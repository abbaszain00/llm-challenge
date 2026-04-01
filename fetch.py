"""
fetch_data.py — Fetch and cache 100 jokes from JokeAPI
--------------------------------------------------------
Fetches exactly 100 unique jokes from https://v2.jokeapi.dev
with all NSFW flags filtered out, and saves them to data/jokes.json.

This script should be run ONCE before ingest.py.
The resulting jokes.json is the static knowledge base for the app.

Usage:
    python fetch_data.py
"""

import json
import time
import requests
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
JOKE_API_URL = "https://v2.jokeapi.dev/joke/Any"
TARGET_COUNT = 100
OUTPUT_PATH = Path("data/jokes.json")

# JokeAPI blacklist flags — filter ALL potentially offensive content
BLACKLIST_FLAGS = "nsfw,religious,political,racist,sexist,explicit"

# JokeAPI supports max 10 jokes per request
BATCH_SIZE = 10


def fetch_jokes(target: int = TARGET_COUNT) -> list[dict]:
    """
    Fetch `target` unique jokes from JokeAPI in batches.
    Deduplicates by joke ID.
    """
    jokes = {}  # id → joke dict, for deduplication
    attempts = 0
    max_attempts = 20  # safety limit

    print(f"🎭 Fetching {target} unique jokes from JokeAPI...")
    print(f"   Blacklisted flags: {BLACKLIST_FLAGS}\n")

    while len(jokes) < target and attempts < max_attempts:
        attempts += 1
        needed = target - len(jokes)
        batch = min(needed, BATCH_SIZE)

        params = {
            "amount": batch,
            "blacklistFlags": BLACKLIST_FLAGS,
            "lang": "en",
        }

        try:
            response = requests.get(JOKE_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"   ⚠ Request failed (attempt {attempts}): {e}")
            time.sleep(2)
            continue

        # API returns either a single joke or {"jokes": [...]} for batches
        raw_jokes = data.get("jokes", [data]) if "jokes" in data else [data]

        new = 0
        for joke in raw_jokes:
            if joke.get("error"):
                print(f"   ⚠ API error: {joke.get('message', 'unknown')}")
                continue

            jid = joke.get("id")
            if jid not in jokes:
                jokes[jid] = normalise(joke)
                new += 1

        print(f"   Batch {attempts}: +{new} jokes | Total: {len(jokes)}/{target}")

        # Be polite to the API
        time.sleep(0.5)

    return list(jokes.values())


def normalise(joke: dict) -> dict:
    """
    Normalise JokeAPI's two formats into one consistent structure:
      - type "single": just a "joke" field
      - type "twopart": "setup" + "delivery"
    """
    base = {
        "id": joke.get("id"),
        "category": joke.get("category", "Unknown"),
        "type": joke.get("type", "single"),
        "flags": joke.get("flags", {}),
        "safe": joke.get("safe", True),
        "lang": joke.get("lang", "en"),
    }

    if joke.get("type") == "twopart":
        base["setup"] = joke.get("setup", "")
        base["delivery"] = joke.get("delivery", "")
        base["full_text"] = f"{joke.get('setup', '')} ... {joke.get('delivery', '')}"
    else:
        base["joke"] = joke.get("joke", "")
        base["full_text"] = joke.get("joke", "")

    return base


def save_jokes(jokes: list[dict], path: Path = OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "total": len(jokes),
        "source": "https://v2.jokeapi.dev/joke/Any",
        "blacklisted_flags": BLACKLIST_FLAGS,
        "jokes": jokes,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(jokes)} jokes to {path}")


def summarise(jokes: list[dict]):
    from collections import Counter
    categories = Counter(j["category"] for j in jokes)
    types = Counter(j["type"] for j in jokes)

    print(f"\n📊 Dataset summary:")
    print(f"   Total jokes: {len(jokes)}")
    print(f"   Categories: {dict(categories)}")
    print(f"   Types: {dict(types)}")
    print(f"   All safe: {all(j.get('safe', True) for j in jokes)}")


if __name__ == "__main__":
    jokes = fetch_jokes(TARGET_COUNT)

    if len(jokes) < TARGET_COUNT:
        print(f"\n⚠ Only fetched {len(jokes)}/{TARGET_COUNT} jokes — API may be rate limiting.")
        print("   Wait a minute and try again, or proceed with what we have.")
    else:
        print(f"\n🎉 Successfully fetched {len(jokes)} unique jokes!")

    save_jokes(jokes)
    summarise(jokes)
    print(f"\n➡  Next step: run `python ingest.py`")