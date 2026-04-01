"""
ingest.py — Data ingestion pipeline
-------------------------------------
Reads jokes.json from /data, converts each joke into a
searchable text chunk, embeds them, and stores in ChromaDB.

Each joke becomes one chunk — no splitting needed since jokes are short.
Metadata (category, type, id) is stored alongside for filtered queries.

Usage:
    python ingest.py
"""

import os
import json
import hashlib
from pathlib import Path
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "jokes"
EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_PATH = Path("./data/jokes.json")


def joke_to_text(joke: dict) -> str:
    """
    Convert a joke dict to a rich text string for embedding.
    Including category and type helps semantic search find
    jokes by topic/theme even when keywords don't match exactly.
    """
    category = joke.get("category", "Unknown")
    joke_type = joke.get("type", "single")

    if joke_type == "twopart":
        text = (
            f"Category: {category}. "
            f"Type: two-part joke. "
            f"Setup: {joke.get('setup', '')} "
            f"Delivery: {joke.get('delivery', '')}"
        )
    else:
        text = (
            f"Category: {category}. "
            f"Type: single joke. "
            f"{joke.get('joke', '')}"
        )

    return text.strip()


def ingest():
    if not DATA_PATH.exists():
        print(f"❌ {DATA_PATH} not found. Run fetch_data.py first.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    jokes = data.get("jokes", [])
    print(f"📂 Loaded {len(jokes)} jokes from {DATA_PATH}")

    # Set up ChromaDB
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Clear and recreate collection for idempotent re-ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑  Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = [], [], []

    for joke in jokes:
        text = joke_to_text(joke)
        joke_id = str(joke.get("id", hashlib.md5(text.encode()).hexdigest()))

        ids.append(joke_id)
        documents.append(text)
        metadatas.append({
            "id": joke.get("id", 0),
            "category": joke.get("category", "Unknown"),
            "type": joke.get("type", "single"),
            "safe": str(joke.get("safe", True)),
            "full_text": joke.get("full_text", text),
        })

    # Upsert in batches of 50
    batch_size = 50
    for start in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[start:start + batch_size],
            documents=documents[start:start + batch_size],
            metadatas=metadatas[start:start + batch_size],
        )
        print(f"   ✅ Indexed batch {start // batch_size + 1} ({min(start + batch_size, len(ids))}/{len(ids)})")

    # Summary
    categories = Counter(j.get("category", "Unknown") for j in jokes)
    print(f"\n🎉 Done! {len(jokes)} jokes indexed into ChromaDB at '{CHROMA_PATH}'")
    print(f"   Collection: '{COLLECTION_NAME}' | Model: '{EMBED_MODEL}'")
    print(f"   Categories: {dict(categories)}")
    print(f"\n➡  Next step: run `python app.py`")


if __name__ == "__main__":
    ingest()