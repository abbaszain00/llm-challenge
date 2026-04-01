import os
import json
import hashlib
from pathlib import Path
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION = "jokes"
EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_PATH = Path("./data/jokes.json")


def joke_to_text(joke):
    # build a text string per joke for embedding
    # including category helps semantic search find jokes by theme
    cat = joke.get("category", "Unknown")
    if joke.get("type") == "twopart":
        return f"Category: {cat}. Setup: {joke.get('setup', '')} Delivery: {joke.get('delivery', '')}"
    else:
        return f"Category: {cat}. {joke.get('joke', '')}"


def ingest():
    if not DATA_PATH.exists():
        print("jokes.json not found - run fetch_data.py first")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    jokes = data.get("jokes", [])
    print(f"Loaded {len(jokes)} jokes")

    # set up chromadb with local embeddings
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # wipe and recreate so re-running is safe
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = [], [], []

    for joke in jokes:
        text = joke_to_text(joke)
        jid = str(joke.get("id", hashlib.md5(text.encode()).hexdigest()))
        ids.append(jid)
        docs.append(text)
        metas.append({
            "id": joke.get("id", 0),
            "category": joke.get("category", "Unknown"),
            "type": joke.get("type", "single"),
            "safe": str(joke.get("safe", True)),
            "full_text": joke.get("full_text", text),
        })

    # upsert in batches
    for i in range(0, len(ids), 50):
        collection.upsert(
            ids=ids[i:i+50],
            documents=docs[i:i+50],
            metadatas=metas[i:i+50],
        )

    cats = Counter(j.get("category", "Unknown") for j in jokes)
    print(f"Indexed {len(jokes)} jokes into ChromaDB")
    print(f"Categories: {dict(cats)}")


if __name__ == "__main__":
    ingest()