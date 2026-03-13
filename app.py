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

st.set_page_config(
    page_title="DDR — Diagnostic Report",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Kill green code badges */
code, pre, kbd,
[data-testid="stMarkdownContainer"] code {
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    background: #f2f2f2 !important;
    color: #444 !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 1px 6px !important;
    font-size: 0.76rem !important;
}

/* Pure white background */
.stApp {
    background: #ffffff !important;
}
.block-container {
    max-width: 800px !important;
    padding: 0 1.5rem 5rem 1.5rem !important;
    background: #ffffff !important;
}

/* ── Hero ── */
.hero {
    padding: 3.5rem 0 2rem 0;
    border-bottom: 1px solid #ebebeb;
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #444;
    margin-bottom: 0.65rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0a0a0a;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
}
.hero-desc {
    font-size: 0.88rem;
    color: #555;
    font-weight: 400;
    font-style: italic;
    margin: 0;
}

/* ── Upload labels ── */
.upload-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #333;
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.upload-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #555;
    display: inline-block;
}

/* File pill */
.file-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.4rem;
    background: #f2f2f2;
    border: 1px solid #ddd;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.76rem;
    color: #333;
    font-weight: 500;
}
.file-pill-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #111;
    flex-shrink: 0;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #fafafa !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 8px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:focus-within {
    border-color: #111 !important;
}

/* ── ALL buttons — black ── */
div.stButton > button,
div.stButton > button[kind="primary"],
div.stButton > button[kind="secondary"] {
    background: #0a0a0a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
}
div.stButton > button:hover {
    background: #222 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* ── Download button — black ── */
div.stDownloadButton > button {
    background: #0a0a0a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 0.75rem 2rem !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
}
div.stDownloadButton > button:hover {
    background: #222 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
}

/* ── Stat row ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #ebebeb;
    border: 1px solid #ebebeb;
    border-radius: 8px;
    overflow: hidden;
    margin: 1.5rem 0 1.25rem 0;
}
.stat-cell {
    background: #fff;
    padding: 1.1rem 0.5rem 0.9rem 0.5rem;
    text-align: center;
}
.stat-num {
    font-size: 1.6rem;
    font-weight: 600;
    color: #0a0a0a;
    letter-spacing: -0.03em;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #777;
    margin-top: 0.3rem;
    font-weight: 600;
}

/* ── Section heading ── */
.section-head {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #333;
    margin: 2rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #ebebeb;
}

/* ── Expanders ── */
details { margin-bottom: 4px !important; }
details summary {
    background: #fff !important;
    border: 1px solid #ebebeb !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #111 !important;
    padding: 0.7rem 1rem !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
details summary:hover { background: #fafafa !important; }
details[open] summary {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
    border-bottom-color: #f2f2f2 !important;
}
details > div {
    background: #fafafa !important;
    border: 1px solid #ebebeb !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
    padding: 0.9rem 1rem !important;
    font-size: 0.85rem !important;
    color: #222 !important;
    line-height: 1.7 !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-size: 0.81rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #fff !important;
    border-right: 1px solid #ebebeb !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.85rem !important;
    color: #333 !important;
    line-height: 1.7 !important;
    font-weight: 400 !important;
}
section[data-testid="stSidebar"] strong {
    color: #0a0a0a !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #777;
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid #ebebeb;
    letter-spacing: 0.03em;
    line-height: 1.8;
    font-weight: 400;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────
with st.sidebar:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    st.markdown("**DDR Generator**")
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("LangChain — Orchestration")
    st.markdown("ChatGroq — Llama 3.3 70B")
    st.markdown("PyMuPDF — PDF extraction")
    st.markdown("ReportLab — PDF output")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("Temperature: 0.1")
    st.markdown("Max tokens: 3,000")
    st.markdown("Format: JSON")
    st.markdown(
        "<div style='margin-top:2.5rem;font-size:0.72rem;color:#555;"
        "text-transform:uppercase;letter-spacing:0.1em;font-weight:600;'>v1.0</div>",
        unsafe_allow_html=True
    )


# ── Hero ─────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Structural Analysis Platform</div>
    <div class="hero-title">Detailed Diagnostic<br>Report</div>
    <div class="hero-desc">LangChain + Llama 3.3 70B — AI-assisted inspection synthesis</div>
</div>
""", unsafe_allow_html=True)


# ── LangChain chain ──────────────────────────
def get_chain(key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=key,
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
                story.append(Paragraph(
                    f"Figure: {img['doc'].upper()} — Page {img['page']}", caption))
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
        story.append(Paragraph(
            f"<b>{labels.get(p, '')} [{p}]</b> {a.get('action', '')}", body))

    story.append(Paragraph("6. Additional Notes", h1))
    story.append(Paragraph(ddr_data.get("additional_notes", "Not Available"), body))

    story.append(Paragraph("7. Missing or Unclear Information", h1))
    story.append(Paragraph(ddr_data.get("missing_or_unclear_info", "Not Available"), body))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2196F3")))
    story.append(Paragraph(
        "AI-generated report. Please verify findings with a qualified inspector.",
        footer_style))
    doc.build(story)


# ── Upload UI ────────────────────────────────
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("""
    <div class="upload-label">
        <span class="upload-dot"></span>Inspection Report
    </div>""", unsafe_allow_html=True)
    inspection_file = st.file_uploader(
        "inspection", type=["pdf"], key="insp", label_visibility="collapsed")
    if inspection_file:
        st.markdown(f"""
        <div class="file-pill">
            <span class="file-pill-dot"></span>
            {inspection_file.name[:28]}{'…' if len(inspection_file.name) > 28 else ''}
            &nbsp;·&nbsp; {round(inspection_file.size/1024, 1)} KB
        </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="upload-label">
        <span class="upload-dot"></span>Thermal Report
    </div>""", unsafe_allow_html=True)
    thermal_file = st.file_uploader(
        "thermal", type=["pdf"], key="therm", label_visibility="collapsed")
    if thermal_file:
        st.markdown(f"""
        <div class="file-pill">
            <span class="file-pill-dot"></span>
            {thermal_file.name[:28]}{'…' if len(thermal_file.name) > 28 else ''}
            &nbsp;·&nbsp; {round(thermal_file.size/1024, 1)} KB
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

# ── Generate ─────────────────────────────────
if st.button("Generate Report", type="primary", use_container_width=True):
    if not api_key:
        st.error("GROQ_API_KEY is not configured. Add it to your .env file or Streamlit secrets.")
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

            # Stats
            n_areas   = len(ddr_data.get("area_wise_observations", []))
            n_risks   = len(ddr_data.get("severity_assessment", []))
            n_actions = len(ddr_data.get("recommended_actions", []))
            n_images  = len(all_images)

            st.markdown(f"""
<div class="stat-row">
    <div class="stat-cell">
        <div class="stat-num">{n_areas}</div>
        <div class="stat-lbl">Areas</div>
    </div>
    <div class="stat-cell">
        <div class="stat-num">{n_risks}</div>
        <div class="stat-lbl">Risk Zones</div>
    </div>
    <div class="stat-cell">
        <div class="stat-num">{n_actions}</div>
        <div class="stat-lbl">Actions</div>
    </div>
    <div class="stat-cell">
        <div class="stat-num">{n_images}</div>
        <div class="stat-lbl">Images</div>
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

            st.markdown('<div class="section-head">Report Preview</div>', unsafe_allow_html=True)

            with st.expander("01 — Property Issue Summary", expanded=True):
                st.write(ddr_data.get("property_issue_summary", "Not Available"))

            with st.expander("02 — Area-wise Observations"):
                for obs in ddr_data.get("area_wise_observations", []):
                    st.markdown(f"**{obs.get('area')}**")
                    st.write(obs.get("observation"))
                    st.divider()

            with st.expander("03 — Probable Root Cause"):
                for item in ddr_data.get("probable_root_cause", []):
                    st.markdown(f"**{item.get('issue')}:** {item.get('cause')}")

            with st.expander("04 — Severity Assessment"):
                import pandas as pd
                sev = ddr_data.get("severity_assessment", [])
                if sev:
                    st.dataframe(pd.DataFrame(sev), use_container_width=True)

            with st.expander("05 — Recommended Actions"):
                for a in ddr_data.get("recommended_actions", []):
                    st.markdown(f"**[{a.get('priority')}]** {a.get('action')}")

            with st.expander("06 — Additional Notes"):
                st.write(ddr_data.get("additional_notes", "Not Available"))

            with st.expander("07 — Missing or Unclear Information"):
                st.write(ddr_data.get("missing_or_unclear_info", "Not Available"))

            if all_images:
                with st.expander(f"08 — Extracted Images ({len(all_images)})"):
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
<div class="footer">
    AI-generated reports are supplementary tools.<br>
    Always verify findings with a qualified structural inspector.
</div>
""", unsafe_allow_html=True)