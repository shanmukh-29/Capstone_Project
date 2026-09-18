# Zepto Data and AI Platform Capstone

This repository contains the three required modules: `data_pipeline`, `analytics`, and `support_assistant`. It uses one consolidated UTF-8 `requirements.txt`.

## Setup

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Run

```powershell
py data_pipeline/pipeline.py
py analytics/analysis.py
uvicorn support_assistant.main:app --reload
```

The data pipeline scrapes and normalizes book data into SQLite using the fixed `1 GBP = 105.50 INR` rate. The analytics pipeline loads Titanic once, creates the committed CSV fallback, documents EDA decisions, trains comparable classification pipelines, tunes an OOB Random Forest, and saves the complete best pipeline. The support assistant provides an offline deterministic policy RAG baseline with validated JSON responses and a Dockerfile; optional real LLM behavior is gated behind `MOCK_LLM=0`.

See [data_pipeline/README.md](data_pipeline/README.md), [analytics/README.md](analytics/README.md), and [support_assistant/README.md](support_assistant/README.md) for module details.
