# 🧠 AI Solution: Unstructured to Structured JSON Converter

This repository contains a modular and scalable AI-powered system that converts unstructured text (e.g., email threads, policy documents, requirement drafts) into structured data adhering to a given JSON schema using Large Language Models (LLMs).

---

## 📌 Problem Statement

In complex B2B workflows, one critical step involves converting free-form textual data into machine-usable formats such as deeply nested JSON. The challenge is heightened when:
- Documents are lengthy (50 pages to 10MB+),
- JSON schemas are multi-level and dynamic (up to 150k tokens),
- Manual extraction is not scalable or reliable.

This solution automates that process using LLMs by extracting structured fields from raw documents based on the schema definition, ensuring schema validation and confidence scoring.

---

## ✅ Features

- 📄 **Supports Large Documents**: Handles context-limited models via intelligent chunking.
- 🧩 **Handles Complex JSON Schemas**: Including nested objects, arrays, enums, and dynamic keys.
- 🧠 **LLM-Driven Extraction**: Uses schema-aware prompts to extract high-quality values.
- 🧪 **Confidence Scoring**: Scores each extracted field to guide human review.
- 🔧 **Schema Validation**: Final output is strictly validated against the input schema.

---

## 🚧 Limitations

- ❌ **No Web UI Yet**: The solution currently runs only via CLI. Future work includes a web interface for uploading documents and reviewing results.
- ⚠️ **Basic Handling of Arrays & Dynamic Keys**: PatternProperties and recursive array items are partially supported.
- 🐢 **Sequential Extraction**: Processing is currently done sequentially for simplicity. Can be parallelized in future.

---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/darshit-pithadia/Metaforms_AI.git

2. Install dependencies:
```bash
pip install -r requirements.txt

3. Set your LLM API key (Groq API example):
```bash
export GROQ_API_KEY=your_api_key_here

Usage
Run the script using the command-line with the required input file and JSON schema:
```bash
python script.py --path/to/document.txt --path/to/schema.json

Output:
A structured JSON output saved locally or printed to the console
Optional log information and confidence scores per field

Future Enhancements
1.Semantic-aware document chunking
2.Parallel field-level LLM extraction
3.Web interface for document upload and result review
4.Structured JSON outputs using LLM’s JSON mode
5.Human-in-the-loop feedback system

Author
Darshit Pithadia
