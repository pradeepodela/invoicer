# bulk_invoice_csv.py
import base64, json, os, tempfile, mimetypes, csv
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from pydantic import BaseModel, Field
from openai import OpenAI
import io

# ------------------------------------------------------------------ #
# 0.  ENV + CLIENTS
# ------------------------------------------------------------------ #
load_dotenv()

@st.cache_resource(show_spinner=False)
def get_mistral() -> Mistral:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        st.error("❌  Set MISTRAL_API_KEY in .env")
        st.stop()
    return Mistral(api_key=key)

@st.cache_resource(show_spinner=False)
def get_openrouter() -> OpenAI | None:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

mistral_client = get_mistral()
openrouter_client = get_openrouter()

# ------------------------------------------------------------------ #
# 1.  INVOICE DATA MODELS
# ------------------------------------------------------------------ #
class LineItem(BaseModel):
    description: str = Field(default="", description="Product/service description")
    quantity: float = Field(default=0.0, description="Quantity ordered")
    unit_price: float = Field(default=0.0, description="Price per unit")
    total_price: float = Field(default=0.0, description="Total price for this line item")
    unit: Optional[str] = Field(default=None, description="Unit of measurement (e.g., 'each', 'kg', 'hours')")
    sku: Optional[str] = Field(default=None, description="Product SKU or code")
    tax_rate: Optional[float] = Field(default=None, description="Tax rate for this item")

class InvoiceData(BaseModel):
    # Invoice Header
    invoice_number: str = Field(default="", description="Invoice number")
    invoice_date: str = Field(default="", description="Invoice date in YYYY-MM-DD format")
    due_date: Optional[str] = Field(default=None, description="Due date in YYYY-MM-DD format")
    
    # Vendor Information
    vendor_name: str = Field(default="", description="Vendor/supplier name")
    vendor_address: Optional[str] = Field(default=None, description="Vendor address")
    vendor_phone: Optional[str] = Field(default=None, description="Vendor phone number")
    vendor_email: Optional[str] = Field(default=None, description="Vendor email")
    vendor_tax_id: Optional[str] = Field(default=None, description="Vendor tax ID")
    
    # Customer Information
    customer_name: Optional[str] = Field(default=None, description="Customer name")
    customer_address: Optional[str] = Field(default=None, description="Customer address")
    customer_phone: Optional[str] = Field(default=None, description="Customer phone")
    customer_email: Optional[str] = Field(default=None, description="Customer email")
    
    # Line Items
    line_items: List[LineItem] = Field(default_factory=list, description="List of invoice line items")
    
    # Totals
    subtotal: float = Field(default=0.0, description="Subtotal amount")
    tax_amount: Optional[float] = Field(default=None, description="Total tax amount")
    discount_amount: Optional[float] = Field(default=0.0, description="Discount amount")
    total_amount: float = Field(default=0.0, description="Total invoice amount")
    
    # Additional fields
    currency: Optional[str] = Field(default="USD", description="Currency code")
    payment_terms: Optional[str] = Field(default=None, description="Payment terms")
    notes: Optional[str] = Field(default=None, description="Additional notes")

# ------------------------------------------------------------------ #
# 2.  OCR HELPERS
# ------------------------------------------------------------------ #
def _merge_md(resp: OCRResponse) -> str:
    md_pages = []
    for p in resp.pages:
        imgs = {i.id: i.image_base64 for i in p.images}
        md = p.markdown
        for iid, b64 in imgs.items():
            md = md.replace(f"![{iid}]({iid})", f"![{iid}]({b64})")
        md_pages.append(md)
    return "\n\n".join(md_pages)

# ------------------------------------------------------------------ #
# 3.  INVOICE EXTRACTION PROMPT
# ------------------------------------------------------------------ #
def _create_invoice_extraction_prompt(ocr_text: str) -> str:
    return f"""
You are an expert invoice data extraction specialist. Extract ALL information from the invoice OCR text below and convert it into structured JSON format.

**CRITICAL INSTRUCTIONS:**
1. **Extract ALL line items** - Don't miss any products/services listed
2. **Be precise with numbers** - Extract exact quantities, prices, and totals as numbers (not strings)
3. **Date format** - Use YYYY-MM-DD format for all dates
4. **Currency** - Remove currency symbols, keep only numeric values
5. **Line item details** - For each item, extract description, quantity, unit price, and total
6. **Vendor/Customer info** - Extract all available contact information
7. **Calculations** - Ensure subtotal, tax, and total amounts are correctly captured
8. **Handle missing data** - Use empty strings for missing text fields, 0.0 for missing numbers, empty arrays for missing lists

**JSON STRUCTURE REQUIRED:**
{{
    "invoice_number": "string",
    "invoice_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD or null",
    "vendor_name": "string",
    "vendor_address": "string or null",
    "vendor_phone": "string or null",
    "vendor_email": "string or null",
    "vendor_tax_id": "string or null",
    "customer_name": "string or null",
    "customer_address": "string or null",
    "customer_phone": "string or null",
    "customer_email": "string or null",
    "line_items": [
        {{
            "description": "string",
            "quantity": number,
            "unit_price": number,
            "total_price": number,
            "unit": "string or null",
            "sku": "string or null",
            "tax_rate": number or null
        }}
    ],
    "subtotal": number,
    "tax_amount": number or null,
    "discount_amount": number,
    "total_amount": number,
    "currency": "string",
    "payment_terms": "string or null",
    "notes": "string or null"
}}

**IMPORTANT:** 
- If a field is not present in the invoice, use appropriate defaults (empty string for text, 0.0 for numbers, null for optional fields)
- For required fields, make reasonable inferences from context
- Pay special attention to line items - extract every single item listed
- Ensure all monetary amounts are numeric (no currency symbols)
- Use 0.0 for missing numeric values, not null

**OCR TEXT:**
{ocr_text}

**RESPONSE FORMAT:**
Return ONLY valid JSON that matches the structure above. Include all line items found in the invoice.
"""

# ------------------------------------------------------------------ #
# 4.  LLM WRAPPERS
# ------------------------------------------------------------------ #
DEFAULT_MODEL = "pixtral-12b-latest"
OPENROUTER_MODEL = "qwen/qwen2.5-vl-72b-instruct"

def _mistral_parse(chunks) -> Dict[str, Any]:
    try:
        chat = mistral_client.chat.parse(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": chunks}],
            response_format=InvoiceData,
            temperature=0,
        )
        return json.loads(chat.choices[0].message.parsed.model_dump_json())
    except Exception as e:
        # If structured parsing fails, try regular chat completion
        st.warning(f"Structured parsing failed, trying regular completion: {str(e)}")
        chat = mistral_client.chat.complete(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": chunks}],
            temperature=0,
        )
        raw_response = chat.choices[0].message.content
        return _extract_json(raw_response)

def _extract_json(text: str) -> dict:
    """Extract the first JSON block from text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError("Failed to extract valid JSON from OpenRouter response.")

def _openrouter_parse(prompt_text, img_url) -> Dict[str, Any]:
    if not openrouter_client:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    content = []
    if img_url:
        content.append({"type": "image_url", "image_url": {"url": img_url}})
    content.append({"type": "text", "text": prompt_text})

    completion = openrouter_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        extra_headers={
            "X-Title": "BulkInvoiceCSV",
            "HTTP-Referer": "https://localhost"
        },
    )
    raw = completion.choices[0].message.content.strip()
    return _extract_json(raw)

# ------------------------------------------------------------------ #
# 5.  OCR PROCESSING
# ------------------------------------------------------------------ #
def ocr_pdf(path: str) -> tuple[str, str]:
    """return merged_markdown, data_url"""
    with open(path, "rb") as f: 
        data = f.read()
    b64 = base64.b64encode(data).decode()
    url = f"data:application/pdf;base64,{b64}"

    resp = mistral_client.ocr.process(
        document=DocumentURLChunk(document_url=url),
        model="mistral-ocr-latest",
        include_image_base64=False,
    )
    return (_merge_md(resp), url)

def ocr_image(uploaded) -> tuple[str, str]:
    data = uploaded.getvalue()
    b64 = base64.b64encode(data).decode()
    mime = mimetypes.guess_type(uploaded.name)[0] or "image/jpeg"
    url = f"data:{mime};base64,{b64}"

    resp = mistral_client.ocr.process(
        document=ImageURLChunk(image_url=url),
        model="mistral-ocr-latest",
        include_image_base64=False,
    )
    return (_merge_md(resp), url)

# ------------------------------------------------------------------ #
# 6.  CSV CONVERSION
# ------------------------------------------------------------------ #
def convert_invoices_to_csv(invoices: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert multiple invoices to a single CSV with line-item level detail"""
    rows = []
    
    for invoice_idx, invoice in enumerate(invoices, 1):
        # Extract invoice header data
        header_data = {
            'file_number': invoice_idx,
            'invoice_number': invoice.get('invoice_number', ''),
            'invoice_date': invoice.get('invoice_date', ''),
            'due_date': invoice.get('due_date', ''),
            'vendor_name': invoice.get('vendor_name', ''),
            'vendor_address': invoice.get('vendor_address', ''),
            'vendor_phone': invoice.get('vendor_phone', ''),
            'vendor_email': invoice.get('vendor_email', ''),
            'vendor_tax_id': invoice.get('vendor_tax_id', ''),
            'customer_name': invoice.get('customer_name', ''),
            'customer_address': invoice.get('customer_address', ''),
            'customer_phone': invoice.get('customer_phone', ''),
            'customer_email': invoice.get('customer_email', ''),
            'subtotal': invoice.get('subtotal', 0),
            'tax_amount': invoice.get('tax_amount', 0),
            'discount_amount': invoice.get('discount_amount', 0),
            'total_amount': invoice.get('total_amount', 0),
            'currency': invoice.get('currency', 'USD'),
            'payment_terms': invoice.get('payment_terms', ''),
            'notes': invoice.get('notes', ''),
        }
        
        # Process line items
        line_items = invoice.get('line_items', [])
        if not line_items:
            # If no line items, create one row with header data
            row = header_data.copy()
            row.update({
                'line_item_number': 1,
                'item_description': '',
                'quantity': 0,
                'unit_price': 0,
                'line_total': 0,
                'unit': '',
                'sku': '',
                'tax_rate': 0,
            })
            rows.append(row)
        else:
            # Create one row per line item
            for item_idx, item in enumerate(line_items, 1):
                row = header_data.copy()
                row.update({
                    'line_item_number': item_idx,
                    'item_description': item.get('description', ''),
                    'quantity': item.get('quantity', 0),
                    'unit_price': item.get('unit_price', 0),
                    'line_total': item.get('total_price', 0),
                    'unit': item.get('unit', ''),
                    'sku': item.get('sku', ''),
                    'tax_rate': item.get('tax_rate', 0),
                })
                rows.append(row)
    
    return pd.DataFrame(rows)

def create_summary_csv(invoices: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary CSV with one row per invoice"""
    summary_rows = []
    
    for invoice_idx, invoice in enumerate(invoices, 1):
        line_items = invoice.get('line_items', [])
        summary_row = {
            'file_number': invoice_idx,
            'invoice_number': invoice.get('invoice_number', ''),
            'invoice_date': invoice.get('invoice_date', ''),
            'due_date': invoice.get('due_date', ''),
            'vendor_name': invoice.get('vendor_name', ''),
            'customer_name': invoice.get('customer_name', ''),
            'line_items_count': len(line_items),
            'subtotal': invoice.get('subtotal', 0),
            'tax_amount': invoice.get('tax_amount', 0),
            'discount_amount': invoice.get('discount_amount', 0),
            'total_amount': invoice.get('total_amount', 0),
            'currency': invoice.get('currency', 'USD'),
        }
        summary_rows.append(summary_row)
    
    return pd.DataFrame(summary_rows)

# ------------------------------------------------------------------ #
# 7.  STREAMLIT UI
# ------------------------------------------------------------------ #
st.set_page_config("Bulk Invoice to CSV", "📄", layout="wide")
st.title("📄 Bulk Invoice to CSV Formatter")
st.markdown("Upload multiple invoice files and convert them to structured CSV format with detailed line items.")
st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    llm_choice = st.selectbox(
        "Select LLM for extraction",
        ["Mistral", "OpenRouter"] if openrouter_client else ["Mistral"],
        help="Choose the LLM to use for invoice data extraction"
    )
    
    if llm_choice == "OpenRouter" and not openrouter_client:
        st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env")
        llm_choice = "Mistral"
    
    st.subheader("📊 Output Options")
    include_detailed_csv = st.checkbox("Generate detailed CSV (line-item level)", value=True)
    include_summary_csv = st.checkbox("Generate summary CSV (invoice level)", value=True)
    
    st.subheader("📋 CSV Column Info")
    with st.expander("Detailed CSV Columns"):
        st.write("""
        - **file_number**: Sequential number for uploaded files
        - **invoice_number**: Invoice number from document
        - **invoice_date**: Invoice date (YYYY-MM-DD)
        - **vendor_name**: Vendor/supplier name
        - **customer_name**: Customer name
        - **line_item_number**: Line item sequence number
        - **item_description**: Product/service description
        - **quantity**: Quantity ordered
        - **unit_price**: Price per unit
        - **line_total**: Total for this line item
        - **subtotal**: Invoice subtotal
        - **tax_amount**: Total tax
        - **total_amount**: Final invoice total
        - And more...
        """)

# Main content
st.subheader("📤 Upload Invoice Files")
uploaded_files = st.file_uploader(
    "Choose invoice files (PDF or images)",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload multiple invoice files to process in batch"
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
    
    process_button = st.button("🚀 Process All Invoices", type="primary")
    
    if process_button:
        all_invoices = []
        processing_status = st.empty()
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            processing_status.text(f"Processing {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")
            
            try:
                # Add detailed error logging
                st.write(f"🔄 Processing: {uploaded_file.name}")
                
                # OCR processing
                if Path(uploaded_file.name).suffix.lower() == ".pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    md, preview_url = ocr_pdf(tmp_path)
                    os.unlink(tmp_path)
                else:
                    md, preview_url = ocr_image(uploaded_file)
                
                st.write(f"✅ OCR completed for {uploaded_file.name}")
                
                # LLM extraction
                prompt = _create_invoice_extraction_prompt(md)
                
                if llm_choice == "Mistral":
                    chunks = [TextChunk(text=prompt)]
                    if preview_url.startswith("data:image"):
                        chunks = [ImageURLChunk(image_url=preview_url)] + chunks
                    invoice_data = _mistral_parse(chunks)
                else:  # OpenRouter
                    img_arg = preview_url if preview_url.startswith("data:image") else None
                    invoice_data = _openrouter_parse(prompt, img_arg)
                
                st.write(f"✅ Data extraction completed for {uploaded_file.name}")
                
                # Validate and clean the extracted data
                if not isinstance(invoice_data, dict):
                    raise ValueError("Invalid data structure returned from LLM")
                
                # Clean numeric fields
                numeric_fields = ['subtotal', 'tax_amount', 'discount_amount', 'total_amount']
                for field in numeric_fields:
                    if field in invoice_data:
                        try:
                            invoice_data[field] = float(invoice_data[field]) if invoice_data[field] is not None else 0.0
                        except (ValueError, TypeError):
                            invoice_data[field] = 0.0
                
                # Clean line items
                if 'line_items' in invoice_data and isinstance(invoice_data['line_items'], list):
                    cleaned_items = []
                    for item in invoice_data['line_items']:
                        if isinstance(item, dict):
                            # Clean numeric fields in line items
                            item_numeric_fields = ['quantity', 'unit_price', 'total_price', 'tax_rate']
                            for field in item_numeric_fields:
                                if field in item:
                                    try:
                                        item[field] = float(item[field]) if item[field] is not None else 0.0
                                    except (ValueError, TypeError):
                                        item[field] = 0.0
                            cleaned_items.append(item)
                    invoice_data['line_items'] = cleaned_items
                
                # Add metadata
                invoice_data['source_file'] = uploaded_file.name
                invoice_data['processed_at'] = datetime.now().isoformat()
                
                all_invoices.append(invoice_data)
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                st.write(f"Error details: {type(e).__name__}: {str(e)}")
                continue
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        processing_status.text("Processing complete!")
        
        if all_invoices:
            st.success(f"✅ Successfully processed {len(all_invoices)} invoice(s)")
            
            # Display results
            st.subheader("📊 Processing Results")
            
            # Show summary statistics
            total_line_items = sum(len(inv.get('line_items', [])) for inv in all_invoices)
            total_amount = sum(inv.get('total_amount', 0) for inv in all_invoices)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Invoices Processed", len(all_invoices))
            col2.metric("Total Line Items", total_line_items)
            col3.metric("Total Amount", f"${total_amount:,.2f}")
            col4.metric("Average per Invoice", f"${total_amount/len(all_invoices):,.2f}")
            
            # Generate CSV files
            csv_files = {}
            
            if include_detailed_csv:
                detailed_df = convert_invoices_to_csv(all_invoices)
                csv_files['detailed'] = detailed_df.to_csv(index=False)
                
                st.subheader("📋 Detailed CSV Preview (Line-Item Level)")
                st.dataframe(detailed_df.head(20), use_container_width=True)
                if len(detailed_df) > 20:
                    st.info(f"Showing first 20 rows of {len(detailed_df)} total rows")
            
            if include_summary_csv:
                summary_df = create_summary_csv(all_invoices)
                csv_files['summary'] = summary_df.to_csv(index=False)
                
                st.subheader("📋 Summary CSV Preview (Invoice Level)")
                st.dataframe(summary_df, use_container_width=True)
            
            # Download buttons
            st.subheader("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            if 'detailed' in csv_files:
                with col1:
                    st.download_button(
                        "📄 Download Detailed CSV",
                        csv_files['detailed'],
                        file_name=f"invoices_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            if 'summary' in csv_files:
                with col2:
                    st.download_button(
                        "📊 Download Summary CSV",
                        csv_files['summary'],
                        file_name=f"invoices_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # Download raw JSON
            with col3:
                st.download_button(
                    "🔧 Download Raw JSON",
                    json.dumps(all_invoices, indent=2),
                    file_name=f"invoices_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # Display individual invoice details
            st.subheader("🔍 Individual Invoice Details")
            
            for idx, invoice in enumerate(all_invoices, 1):
                with st.expander(f"Invoice {idx}: {invoice.get('invoice_number', 'N/A')} - {invoice.get('vendor_name', 'N/A')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Invoice Info:**")
                        st.write(f"Number: {invoice.get('invoice_number', 'N/A')}")
                        st.write(f"Date: {invoice.get('invoice_date', 'N/A')}")
                        st.write(f"Vendor: {invoice.get('vendor_name', 'N/A')}")
                        st.write(f"Customer: {invoice.get('customer_name', 'N/A')}")
                        st.write(f"Total: ${invoice.get('total_amount', 0):,.2f}")
                    
                    with col2:
                        st.write("**Line Items:**")
                        line_items = invoice.get('line_items', [])
                        if line_items:
                            for i, item in enumerate(line_items, 1):
                                st.write(f"{i}. {item.get('description', 'N/A')} - Qty: {item.get('quantity', 0)} - ${item.get('total_price', 0):,.2f}")
                        else:
                            st.write("No line items found")
        
        else:
            st.error("❌ No invoices were successfully processed")

else:
    st.info("👆 Please upload invoice files to get started")
    
    # Show example of expected output
    st.subheader("📋 Example Output Structure")
    
    example_data = {
        "file_number": 1,
        "invoice_number": "INV-2025-001",
        "invoice_date": "2025-01-15",
        "vendor_name": "ABC Supply Co.",
        "customer_name": "XYZ Company",
        "line_item_number": 1,
        "item_description": "Office Supplies",
        "quantity": 5,
        "unit_price": 25.00,
        "line_total": 125.00,
        "subtotal": 250.00,
        "tax_amount": 20.00,
        "total_amount": 270.00
    }
    
    st.json(example_data)
