import streamlit as st
import fitz
import os
import tempfile
from io import BytesIO
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_CENTER

load_dotenv()

# ── Page config ──────────────────────────────
st.set_page_config(page_title="DDR Report Generator", layout="wide")

# ── Minimal CSS ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #f9f9f8; }

.block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

.page-header {
    margin-bottom: 2rem;
    border-bottom: 1px solid #e2e2de;
    padding-bottom: 1.25rem;
}
.page-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #111;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.page-header p {
    font-size: 0.8rem;
    color: #999;
    margin: 0;
    font-weight: 400;
}

.field-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #aaa;
    margin-bottom: 0.4rem;
}

.file-meta {
    font-size: 0.76rem;
    color: #bbb;
    margin-top: 0.35rem;
}

.metrics-row {
    display: flex;
    gap: 1px;
    background: #e2e2de;
    border: 1px solid #e2e2de;
    border-radius: 6px;
    overflow: hidden;
    margin: 1.25rem 0;
}
.metric-item {
    flex: 1;
    background: #fff;
    padding: 1rem;
    text-align: center;
}
.metric-num {
    font-size: 1.5rem;
    font-weight: 600;
    color: #111;
    line-height: 1;
}
.metric-lbl {
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #bbb;
    margin-top: 0.25rem;
}

.stButton > button[kind="primary"] {
    background: #111;
    color: #fff;
    border: none;
    border-radius: 5px;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    transition: background 0.2s;
}
.stButton > button[kind="primary"]:hover { background: #333; }

.stDownloadButton > button {
    background: #fff;
    color: #111;
    border: 1px solid #d0d0cc;
    border-radius: 5px;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    transition: border-color 0.2s;
}
.stDownloadButton > button:hover { border-color: #888; }

.streamlit-expanderHeader {
    background: #fff !important;
    border: 1px solid #e2e2de !important;
    border-radius: 5px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: #111 !important;
}
.streamlit-expanderContent {
    background: #fafaf9 !important;
    border: 1px solid #e2e2de !important;
    border-top: none !important;
    border-radius: 0 0 5px 5px !important;
}

section[data-testid="stSidebar"] {
    background: #fff;
    border-right: 1px solid #e2e2de;
}

[data-testid="stFileUploader"] {
    border: 1px dashed #d0d0cc;
    border-radius: 5px;
    background: #fff;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Detailed Diagnostic Report</h1>
    <p>Structural inspection analysis — LangChain + Llama 3.3 70B</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")
        
    st.markdown("**DDR Generator**")
    st.markdown("<hr style='border:none;border-top:1px solid #e2e2de;margin:0.6rem 0'>",
                unsafe_allow_html=True)
    st.markdown("""
**Stack**

`LangChain` — Orchestration  
`ChatGroq` — Llama 3.3 70B  
`PyMuPDF` — PDF extraction  
`ReportLab` — PDF output

---

**Model**

Temperature: `0.1`  
Max tokens: `3,000`  
Format: `JSON`
    """)
    st.markdown("""
<div style="margin-top:3rem;font-size:0.7rem;color:#ccc;text-transform:uppercase;letter-spacing:0.08em;">
v1.0
</div>
""", unsafe_allow_html=True)


# ── LangChain chain ──────────────────────────
def get_chain(api_key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,
        max_tokens=3000
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert building inspection report writer.
Generate a structured DDR (Detailed Diagnostic Report) from inspection data.

STRICT RULES:
- Do NOT invent facts not present in the documents
- If info is missing use "Not Available"
- If info conflicts mention the conflict explicitly
- Use simple client-friendly language
- Avoid unnecessary technical jargon
- Do not duplicate observations
- Return ONLY valid JSON no markdown no extra text"""),
        ("human", """
=== INSPECTION REPORT ===
{inspection_text}

=== THERMAL REPORT ===
{thermal_text}

=== IMAGES FOUND IN DOCUMENTS ===
{image_list}

Generate JSON with EXACTLY this structure:
{{
  "property_issue_summary": "2-3 sentence overall summary",
  "area_wise_observations": [
    {{"area": "Area name", "observation": "Detailed observation", "related_image_ids": ["image_id_here"]}}
  ],
  "probable_root_cause": [
    {{"issue": "Issue name", "cause": "Root cause explanation"}}
  ],
  "severity_assessment": [
    {{"area": "Area name", "severity": "High / Medium / Low", "reasoning": "Why this severity"}}
  ],
  "recommended_actions": [
    {{"priority": "Immediate / Short-term / Long-term", "action": "Action description"}}
  ],
  "additional_notes": "Extra observations or context",
  "missing_or_unclear_info": "Missing or conflicting info or Not Available"
}}""")
    ])
    return prompt | llm | JsonOutputParser()


# ── PDF extraction ───────────────────────────
def extract_pdf_content(pdf_bytes, doc_label="doc"):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        full_text += f"\n\n--- PAGE {page_num + 1} ---\n{page_text}"
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                if len(img_bytes) < 1000:
                    continue
                images.append({
                    "page": page_num + 1,
                    "index": img_idx,
                    "bytes": img_bytes,
                    "ext": base_image["ext"],
                    "doc": doc_label,
                    "id": f"{doc_label}_p{page_num+1}_i{img_idx}",
                    "page_text": page_text
                })
            except Exception:
                continue
    doc.close()
    return full_text, images


# ── PDF builder ──────────────────────────────
def build_pdf(ddr_data, all_images, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13,
                        textColor=colors.HexColor("#16213e"),
                        backColor=colors.HexColor("#e8f4fd"),
                        spaceBefore=14, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11,
                        textColor=colors.HexColor("#0f3460"), spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)
    caption = ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8,
                              textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8)
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                                   textColor=colors.grey, alignment=TA_CENTER)
    story = []

    story.append(Paragraph("DDR — Detailed Diagnostic Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", caption))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2196F3")))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("1. Property Issue Summary", h1))
    story.append(Paragraph(ddr_data.get("property_issue_summary", "Not Available"), body))

    story.append(Paragraph("2. Area-wise Observations", h1))
    used_image_ids = set()
    for obs in ddr_data.get("area_wise_observations", []):
        area = obs.get("area", "")
        observation = obs.get("observation", "Not Available")
        story.append(Paragraph(f"Area: {area}", h2))
        story.append(Paragraph(observation, body))
        keywords = [w for w in (area + " " + observation).lower().split() if len(w) > 3]
        matched = [img for img in all_images
                   if img["id"] not in used_image_ids
                   and any(kw in img.get("page_text", "").lower() for kw in keywords)]
        added = 0
        for img in matched[:2]:
            try:
                pil = Image.open(BytesIO(img["bytes"])).convert("RGB")
                if max(pil.size) > 800:
                    pil.thumbnail((800, 800))
                buf = BytesIO()
                pil.save(buf, format="JPEG")
                buf.seek(0)
                story.append(RLImage(buf, width=12*cm, height=8*cm))
                story.append(Paragraph(f"Figure: {img['doc'].upper()} — Page {img['page']}", caption))
                used_image_ids.add(img["id"])
                added += 1
            except Exception:
                story.append(Paragraph("Image Not Available", caption))
        if added == 0:
            story.append(Paragraph("Image Not Available", caption))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))

    story.append(Paragraph("3. Probable Root Cause", h1))
    for item in ddr_data.get("probable_root_cause", []):
        story.append(Paragraph(
            f"<b>{item.get('issue', '')}:</b> {item.get('cause', 'Not Available')}", body))

    story.append(Paragraph("4. Severity Assessment", h1))
    rows = [["Area", "Severity", "Reasoning"]]
    for s in ddr_data.get("severity_assessment", []):
        rows.append([s.get("area", ""), s.get("severity", ""), s.get("reasoning", "")])
    if len(rows) > 1:
        tbl = Table(rows, colWidths=[4*cm, 3*cm, 10*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("5. Recommended Actions", h1))
    labels = {"Immediate": "[URGENT]", "Short-term": "[SOON]", "Long-term": "[PLANNED]"}
    for a in ddr_data.get("recommended_actions", []):
        p = a.get("priority", "")
        story.append(Paragraph(f"<b>{labels.get(p, '')} [{p}]</b> {a.get('action', '')}", body))

    story.append(Paragraph("6. Additional Notes", h1))
    story.append(Paragraph(ddr_data.get("additional_notes", "Not Available"), body))

    story.append(Paragraph("7. Missing or Unclear Information", h1))
    story.append(Paragraph(ddr_data.get("missing_or_unclear_info", "Not Available"), body))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2196F3")))
    story.append(Paragraph(
        "AI-generated report. Please verify findings with a qualified inspector.", footer_style))
    doc.build(story)


# ── Upload UI ────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="field-label">Inspection Report</div>', unsafe_allow_html=True)
    inspection_file = st.file_uploader(
        "inspection", type=["pdf"], key="insp", label_visibility="collapsed")
    if inspection_file:
        st.markdown(
            f'<div class="file-meta">{inspection_file.name} &nbsp;·&nbsp; {round(inspection_file.size/1024, 1)} KB</div>',
            unsafe_allow_html=True)

with col2:
    st.markdown('<div class="field-label">Thermal Report</div>', unsafe_allow_html=True)
    thermal_file = st.file_uploader(
        "thermal", type=["pdf"], key="therm", label_visibility="collapsed")
    if thermal_file:
        st.markdown(
            f'<div class="file-meta">{thermal_file.name} &nbsp;·&nbsp; {round(thermal_file.size/1024, 1)} KB</div>',
            unsafe_allow_html=True)

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

if st.button("Generate Report", type="primary", use_container_width=True):
    if not api_key:
        st.error("GROQ_API_KEY is not configured. Set it in your environment or Streamlit secrets.")
    elif not inspection_file or not thermal_file:
        st.error("Please upload both the inspection and thermal PDF documents to proceed.")
    else:
        try:
            with st.status("Extracting document content...", expanded=True) as status:
                st.write("Reading inspection report...")
                insp_text, insp_imgs = extract_pdf_content(inspection_file.read(), "inspection")
                st.write(f"Inspection: {len(insp_text):,} characters, {len(insp_imgs)} images found.")
                st.write("Reading thermal report...")
                therm_text, therm_imgs = extract_pdf_content(thermal_file.read(), "thermal")
                st.write(f"Thermal: {len(therm_text):,} characters, {len(therm_imgs)} images found.")
                all_images = insp_imgs + therm_imgs
                status.update(label="Extraction complete.", state="complete")

            with st.status("Running diagnostic analysis...", expanded=True) as status:
                img_list = "\n".join([
                    f"- ID: {img['id']} | Doc: {img['doc']} | Page: {img['page']}"
                    for img in all_images
                ])
                chain = get_chain(api_key)
                ddr_data = chain.invoke({
                    "inspection_text": insp_text[:4000],
                    "thermal_text": therm_text[:3000],
                    "image_list": img_list[:1000]
                })
                status.update(label="Analysis complete.", state="complete")

            with st.status("Compiling PDF...", expanded=True) as status:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = tmp.name
                build_pdf(ddr_data, all_images, tmp_path)
                status.update(label="PDF ready.", state="complete")

            # Metrics
            n_areas   = len(ddr_data.get("area_wise_observations", []))
            n_risks   = len(ddr_data.get("severity_assessment", []))
            n_actions = len(ddr_data.get("recommended_actions", []))
            n_images  = len(all_images)
            st.markdown(f"""
<div class="metrics-row">
    <div class="metric-item">
        <div class="metric-num">{n_areas}</div>
        <div class="metric-lbl">Areas</div>
    </div>
    <div class="metric-item">
        <div class="metric-num">{n_risks}</div>
        <div class="metric-lbl">Risk Zones</div>
    </div>
    <div class="metric-item">
        <div class="metric-num">{n_actions}</div>
        <div class="metric-lbl">Actions</div>
    </div>
    <div class="metric-item">
        <div class="metric-num">{n_images}</div>
        <div class="metric-lbl">Images</div>
    </div>
</div>
""", unsafe_allow_html=True)

            with open(tmp_path, "rb") as f:
                st.download_button(
                    "Download Report (PDF)",
                    data=f.read(),
                    file_name=f"DDR_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            st.markdown(
                "<hr style='border:none;border-top:1px solid #e2e2de;margin:1.5rem 0'>",
                unsafe_allow_html=True)
            st.markdown('<div class="field-label" style="margin-bottom:0.75rem">Report Preview</div>',
                        unsafe_allow_html=True)

            with st.expander("1. Property Issue Summary", expanded=True):
                st.write(ddr_data.get("property_issue_summary", "Not Available"))

            with st.expander("2. Area-wise Observations"):
                for obs in ddr_data.get("area_wise_observations", []):
                    st.markdown(f"**{obs.get('area')}**")
                    st.write(obs.get("observation"))
                    st.divider()

            with st.expander("3. Probable Root Cause"):
                for item in ddr_data.get("probable_root_cause", []):
                    st.markdown(f"**{item.get('issue')}:** {item.get('cause')}")

            with st.expander("4. Severity Assessment"):
                import pandas as pd
                sev = ddr_data.get("severity_assessment", [])
                if sev:
                    st.dataframe(pd.DataFrame(sev), use_container_width=True)

            with st.expander("5. Recommended Actions"):
                for a in ddr_data.get("recommended_actions", []):
                    st.markdown(f"**[{a.get('priority')}]** {a.get('action')}")

            with st.expander("6. Additional Notes"):
                st.write(ddr_data.get("additional_notes", "Not Available"))

            with st.expander("7. Missing or Unclear Information"):
                st.write(ddr_data.get("missing_or_unclear_info", "Not Available"))

            if all_images:
                with st.expander(f"Extracted Images ({len(all_images)})"):
                    cols = st.columns(3)
                    for i, img in enumerate(all_images):
                        with cols[i % 3]:
                            try:
                                st.image(
                                    Image.open(BytesIO(img["bytes"])),
                                    caption=f"{img['doc'].capitalize()} — Page {img['page']}",
                                    use_column_width=True
                                )
                            except Exception:
                                st.write("Image could not be rendered.")

            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")
            st.exception(e)

# ── Footer ───────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;padding-top:1.25rem;border-top:1px solid #e2e2de;
            font-size:0.72rem;color:#bbb;text-align:center;letter-spacing:0.03em;">
    AI-generated reports are supplementary. Always verify with a qualified inspector.
</div>
""", unsafe_allow_html=True)