import json
import time
import requests
from pathlib import Path

API_URL = "https://v2.jokeapi.dev/joke/Any"
OUTPUT = Path("data/jokes.json")
BLACKLIST = "nsfw,religious,political,racist,sexist,explicit"
BATCH_SIZE = 10
TARGET = 100


def normalise(joke):
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
        base["full_text"] = f"{joke['setup']} ... {joke['delivery']}"
    else:
        base["joke"] = joke.get("joke", "")
        base["full_text"] = joke.get("joke", "")
    return base


def fetch_jokes():
    jokes = {}
    attempts = 0

    print(f"Fetching {TARGET} jokes from JokeAPI...")

    while len(jokes) < TARGET and attempts < 20:
        attempts += 1
        needed = min(TARGET - len(jokes), BATCH_SIZE)

        try:
            r = requests.get(API_URL, params={
                "amount": needed,
                "blacklistFlags": BLACKLIST,
                "lang": "en",
            }, timeout=10)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}, retrying...")
            time.sleep(2)
            continue

        raw = data.get("jokes", [data]) if "jokes" in data else [data]

        for joke in raw:
            if joke.get("error"):
                continue
            jid = joke.get("id")
            if jid not in jokes:
                jokes[jid] = normalise(joke)

        print(f"  batch {attempts}: {len(jokes)}/{TARGET}")
        time.sleep(0.5)

    return list(jokes.values())


if __name__ == "__main__":
    jokes = fetch_jokes()

    if len(jokes) < TARGET:
        print(f"Warning: only got {len(jokes)} jokes, API may be rate limiting")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(jokes),
            "source": API_URL,
            "blacklisted_flags": BLACKLIST,
            "jokes": jokes,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(jokes)} jokes to {OUTPUT}")

    from collections import Counter
    cats = Counter(j["category"] for j in jokes)
    print(f"Categories: {dict(cats)}")