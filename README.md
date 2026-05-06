# Resident Experience + Operations Copilot

Local AI system for property/community managers to analyze resident feedback, prioritize operations, and generate recommendations.

## Step-by-step build plan

1. **Project scaffold**
   - Baseline repository structure
   - Streamlit starter app
   - Dependency setup
2. **Data ingestion**
   - Upload CSV/TXT/PDF
   - Normalize into a unified schema
3. **Classification + urgency scoring**
   - Category classifier
   - Sentiment analysis
   - Priority ranking
4. **RAG question answering**
   - Embeddings + vector store
   - Retrieval and grounded response generation
5. **Dashboard + weekly summary (current step)**
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

## Step 3 demo

After ingestion, the app now also:
- Classifies each record into an issue category
- Estimates sentiment (positive/neutral/negative)
- Assigns urgency (`low`, `medium`, `high`, `urgent`)
- Ranks records by urgency to create a manager priority queue
- Saves classified output to `data/processed/<filename>_classified_normalized.csv`

## Step 4 demo

The app now includes an "Ask your data" panel that:
- Builds a local retrieval index from classified feedback
- Retrieves top relevant evidence for a manager question
- Returns a grounded answer with recommendations
- Optionally uses local `Ollama` for generated responses
- Falls back to deterministic summarization when no LLM is available

## Step 5 demo

The app now adds operations reporting:
- Auto-generated weekly manager summary paragraph
- Recurring issue detection (repeated complaint patterns)
- Action recommendation list based on urgency, category concentration, and recurrence
- Existing analytics and RAG outputs unified into one manager workflow

## Repository layout

- `app/` Streamlit UI
- `src/` backend logic and pipelines
- `data/` local data (ignored except placeholders)
- `docs/` architecture notes and slide assets
- `tests/` test suite
