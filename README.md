# MUJ-DS-23FE10CDS00529

## Student Details

| Field | Details |
|---|---|
| **Name** | Soumil Ahuja |
| **Registration Number** | 23FE10CDS00529 |
| **Branch** | Data Science |
| **Batch** | G |
| **GitHub Username** | soumilahuja |
| **Training Program** | NLP Capstone Training Program |
| **Project Title** | ReviewLens – LLM-Powered Customer Review Analyzer |

## Project Overview

ReviewLens is an NLP project that uses an LLM through API calls (Google Gemini) to turn raw customer reviews into structured insights: sentiment, estimated rating, emotion, aspect-level sentiment, key issue, urgency, a suggested company reply, and an executive summary for a product manager.

Full documentation, setup instructions and the prompt and configuration files are in [`code/reviewlens`](code/reviewlens).

## Repository Structure

```
MUJ-DS-23FE10CDS00529/
├── README.md
├── assignments/
├── notebooks/
├── code/
│   └── reviewlens/        # NLP project (source code, prompt file, config file)
├── resources/
├── presentations/
└── capstone/
```

## Quick Start

```bash
cd code/reviewlens
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# create a .env file with: GEMINI_API_KEY=your-key
python main.py
```