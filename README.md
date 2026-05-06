# Resident Experience + Operations Copilot

Local AI system for property/community managers to analyze resident feedback, prioritize operations, and generate recommendations.

## Step-by-step build plan

1. **Project scaffold**
   - Baseline repository structure
   - Streamlit starter app
   - Dependency setup
2. **Data ingestion (current step)**
   - Upload CSV/TXT/PDF
   - Normalize into a unified schema
3. **Classification + urgency scoring**
   - Category classifier
   - Sentiment analysis
   - Priority ranking
4. **RAG question answering**
   - Embeddings + vector store
   - Retrieval and grounded response generation
5. **Dashboard + weekly summary**
   - Ops trends, urgent queue, recurring issues
   - Auto-generated manager summary
6. **Hardening**
   - Logging/metrics
   - Prompt safety + basic PII masking

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

## Step 2 demo

Run the app, then upload `data/sample_resident_feedback.csv`.

The ingestion flow will:
- Load source data
- Normalize to the unified schema
- Show warnings for missing/invalid fields
- Save output to `data/processed/<filename>_normalized.csv`

## Repository layout

- `app/` Streamlit UI
- `src/` backend logic and pipelines
- `data/` local data (ignored except placeholders)
- `docs/` architecture notes and slide assets
- `tests/` test suite
