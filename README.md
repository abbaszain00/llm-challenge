# LLM Challenge — RAG Application

A grounded LLM application using Retrieval-Augmented Generation (RAG).
Built with Flask, ChromaDB, sentence-transformers, and OpenRouter (Gemini).

---

## Stack

| Component | Technology |
|-----------|------------|
| Web framework | Flask |
| Vector store | ChromaDB (local, persistent) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, runs locally) |
| LLM | `google/gemini-2.0-flash-001` via OpenRouter |
| Env management | `uv` |
| Container | Docker + Docker Compose |

---

## Setup (first time)

### 1. Install uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create environment and install dependencies

```bash
cd llm-challenge
uv venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

uv pip install -r pyproject.toml
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 4. Add your data

Drop your dataset files into the `/data` directory.
Supported formats: `.txt`, `.md`, `.pdf`, `.csv`, `.json`

### 5. Ingest data into ChromaDB

```bash
python ingest.py
```

This chunks your files, embeds them, and stores them locally. Run this once
(or again whenever you change your data).

### 6. Run the app

```bash
python app.py
```

Open http://localhost:5000 in your browser.

---

## Running with Docker

```bash
# Build and start
docker compose up --build

# Run in background
docker compose up --build -d

# Stop
docker compose down
```

**Note:** Run `python ingest.py` before Docker if you want the ChromaDB
to be populated. The `chroma_db/` folder is mounted as a volume.

---

## Evaluation

```bash
# Run default test prompts
python eval.py

# Verbose (prints full answers)
python eval.py --verbose

# Load custom test prompts from JSON
python eval.py --prompts ./my_prompts.json

# Save results to file
python eval.py --save
```

Custom prompt JSON format:
```json
[
  {
    "query": "What is X?",
    "expected_keywords": ["x", "related_term"],
    "should_answer": true
  }
]
```

---

## Project Structure

```
llm-challenge/
├── app.py              # Flask app (routes + session history)
├── ingest.py           # Data ingestion → ChromaDB
├── rag.py              # RAG pipeline (retrieve + generate)
├── eval.py             # Evaluation script
├── data/               # Drop dataset files here
│   └── sample.txt      # Placeholder — replace with real data
├── templates/
│   └── index.html      # Chat UI
├── chroma_db/          # Auto-created by ingest.py (gitignored)
├── pyproject.toml      # uv dependencies
├── .env.example        # API key template
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## On Challenge Day

1. Read the challenge instructions carefully
2. Drop the dataset into `/data` (replace or add to `sample.txt`)
3. Run `python ingest.py` — takes under a minute
4. Test with `python eval.py` using the public test prompts
5. Submit

If the dataset needs scraping or API fetching, a `fetch_data.py` script
can be added quickly — the ingest pipeline accepts any file in `/data`.
