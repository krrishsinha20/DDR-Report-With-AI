import streamlit as st
import fitz  # PyMuPDF
import os
import json
import tempfile
from io import BytesIO
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# PDF generation
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

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="DDR Report Generator", page_icon="🏗️", layout="wide")
st.title("🏗️ DDR Report Generator")
st.caption("AI-powered Detailed Diagnostic Report — Powered by LangChain + Llama")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    
    st.markdown("""
**Stack:**
- `LangChain` — AI orchestration
- `Model` — LLM (llama-3.3-70b)
- `PyMuPDF` — PDF text + image extract
- `ReportLab` — Final PDF output
    """)

# ─────────────────────────────────────────────
# LANGCHAIN CHAIN SETUP
# ─────────────────────────────────────────────
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
    {{
      "area": "Area name",
      "observation": "Detailed observation",
      "related_image_ids": ["image_id_here"]
    }}
  ],
  "probable_root_cause": [
    {{
      "issue": "Issue name",
      "cause": "Root cause explanation"
    }}
  ],
  "severity_assessment": [
    {{
      "area": "Area name",
      "severity": "High / Medium / Low",
      "reasoning": "Why this severity"
    }}
  ],
  "recommended_actions": [
    {{
      "priority": "Immediate / Short-term / Long-term",
      "action": "Action description"
    }}
  ],
  "additional_notes": "Extra observations or context",
  "missing_or_unclear_info": "Missing or conflicting info or Not Available"
}}""")
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain


# ─────────────────────────────────────────────
# HELPER: Extract text + images from PDF
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
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

    image_map = {img["id"]: img for img in all_images}
    story = []

    # Header
    story.append(Paragraph("DDR — Detailed Diagnostic Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", caption))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2196F3")))
    story.append(Spacer(1, 0.3*cm))

    # 1. Summary
    story.append(Paragraph("1. Property Issue Summary", h1))
    story.append(Paragraph(ddr_data.get("property_issue_summary", "Not Available"), body))

    # 2. Area-wise Observations
    story.append(Paragraph("2. Area-wise Observations", h1))
    used_image_ids = set()

    for obs in ddr_data.get("area_wise_observations", []):
        area = obs.get('area', '')
        observation = obs.get("observation", "Not Available")
        story.append(Paragraph(f"Area: {area}", h2))
        story.append(Paragraph(observation, body))

        # Keyword match — area + observation words ko page_text mein dhundo
        keywords = [w for w in (area + " " + observation).lower().split() if len(w) > 3]

        matched = []
        for img in all_images:
            if img["id"] in used_image_ids:
                continue
            page_text_lower = img.get("page_text", "").lower()
            if any(kw in page_text_lower for kw in keywords):
                matched.append(img)

        added = 0
        for img in matched[:2]:  # max 2 images per section
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

    # 3. Root Cause
    story.append(Paragraph("3. Probable Root Cause", h1))
    for item in ddr_data.get("probable_root_cause", []):
        story.append(Paragraph(
            f"<b>{item.get('issue', '')}:</b> {item.get('cause', 'Not Available')}", body))

    # 4. Severity Table
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

    # 5. Recommended Actions
    story.append(Paragraph("5. Recommended Actions", h1))
    icons = {"Immediate": "[URGENT]", "Short-term": "[SOON]", "Long-term": "[PLANNED]"}
    for a in ddr_data.get("recommended_actions", []):
        p = a.get("priority", "")
        story.append(Paragraph(f"<b>{icons.get(p, '')} [{p}]</b> {a.get('action', '')}", body))

    # 6. Additional Notes
    story.append(Paragraph("6. Additional Notes", h1))
    story.append(Paragraph(ddr_data.get("additional_notes", "Not Available"), body))

    # 7. Missing Info
    story.append(Paragraph("7. Missing or Unclear Information", h1))
    story.append(Paragraph(ddr_data.get("missing_or_unclear_info", "Not Available"), body))

    # Footer
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2196F3")))
    story.append(Paragraph(
        "AI-generated report. Please verify findings with a qualified inspector.", footer_style))

    doc.build(story)


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Inspection Report")
    inspection_file = st.file_uploader("Upload PDF", type=["pdf"], key="insp")
with col2:
    st.subheader("🌡️ Thermal Report")
    thermal_file = st.file_uploader("Upload PDF", type=["pdf"], key="therm")

st.markdown("---")

if st.button("🚀 Generate DDR Report", type="primary", use_container_width=True):
    if not api_key:
        st.error("Groq API key chahiye!")
    elif not inspection_file or not thermal_file:
        st.error("Dono PDFs upload karo!")
    else:
        try:
            # Step 1: Extract
            with st.status("📖 PDFs se content extract ho raha hai...", expanded=True) as status:
                st.write("Inspection Report padh raha hoon...")
                insp_text, insp_imgs = extract_pdf_content(inspection_file.read(), "inspection")
                st.write(f"✅ Inspection: {len(insp_text)} chars, {len(insp_imgs)} images")

                st.write("Thermal Report padh raha hoon...")
                therm_text, therm_imgs = extract_pdf_content(thermal_file.read(), "thermal")
                st.write(f"✅ Thermal: {len(therm_text)} chars, {len(therm_imgs)} images")

                all_images = insp_imgs + therm_imgs
                st.write(f"Total images: {len(all_images)}")
                status.update(label="✅ Extraction done!", state="complete")

            # Step 2: LangChain DDR
            with st.status("🤖 Model is generating the DDR PDF", expanded=True) as status:
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
                st.write("✅ DDR generate ho gaya!")
                status.update(label="✅ DDR ready!", state="complete")

            # Step 3: PDF
            with st.status("📝 PDF ban raha hai...", expanded=True) as status:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = tmp.name
                build_pdf(ddr_data, all_images, tmp_path)
                status.update(label="✅ PDF ready!", state="complete")

            # Download button
            st.success("DDR Report Ready !")
            with open(tmp_path, "rb") as f:
                st.download_button(
                    "📥 Download DDR Report (PDF)",
                    data=f.read(),
                    file_name=f"DDR_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            # Preview
            st.markdown("---")
            st.subheader("📋 Report Preview")
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
            with st.expander("7. Missing or Unclear Info"):
                st.write(ddr_data.get("missing_or_unclear_info", "Not Available"))

            if all_images:
                with st.expander(f"📸 Extracted Images ({len(all_images)})"):
                    cols = st.columns(3)
                    for i, img in enumerate(all_images):
                        with cols[i % 3]:
                            try:
                                st.image(Image.open(BytesIO(img["bytes"])),
                                         caption=f"{img['doc']} | Page {img['page']}",
                                         use_column_width=True)
                            except Exception:
                                st.write("Image display nahi ho sakti")

            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
