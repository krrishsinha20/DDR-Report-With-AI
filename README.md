# AI DDR Report Generator

An AI-powered system that converts **Inspection Reports** and **Thermal Reports** into a structured **Detailed Diagnostic Report (DDR)**.

The system extracts observations and images from the provided documents and uses an LLM to generate a clear, client-ready report.

---

# Objective

This project demonstrates how AI can automate the generation of diagnostic reports from technical inspection data.

The system:

* Extracts relevant observations from inspection and thermal documents
* Combines information logically
* Avoids duplicate observations
* Handles missing or conflicting data
* Generates a structured DDR report
* Places extracted images under the relevant observation sections

---

# AI Workflow

Inspection PDF + Thermal PDF
↓
Text Extraction (PyMuPDF)
Image Extraction
↓
LLM Analysis (llama-3.3-70b + LangChain)
↓
Structured DDR Report Generation
↓
Final Client-Ready DDR PDF

---

# Features

* Upload **Inspection Report PDF**
* Upload **Thermal Report PDF**
* Extract text and images automatically
* AI-based report structuring
* Duplicate observation handling
* Missing information detection
* Conflict detection between reports
* Generate **client-friendly DDR report**
* Export final report as **PDF**

---

# Tech Stack

* Python
* Streamlit (Frontend)
* LangChain
* Groq LLM
* llama-3.3-70b
* PyMuPDF
* ReportLab
* Pillow

---

# Project Structure

```
DDR-Report-With-AI
├── app.py
├── requirements.txt
```

---

# How to Run Locally

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit app:

```
streamlit run app.py
```

---

# Deployment

The application can be deployed easily using **Streamlit Cloud**.

---

# Limitations

* Image-to-area mapping depends on document structure
* Some reports may contain ambiguous observations
* Complex layouts may require additional document parsing

---

# Future Improvements

* Advanced document layout detection
* Better image-to-observation linking
* Support for additional inspection report formats
* Improved multimodal reasoning

---

# Demo

---

# Author

Krrish Sinha
