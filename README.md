# llm-challenge

Joke chatbot grounded on 100 jokes fetched from JokeAPI. Built with Flask and OpenRouter (Gemini).

The bot only answers questions about its joke database — it won't generate jokes, answer off-topic questions, or serve NSFW content.

## Stack

- Flask (web server)
- OpenRouter → `google/gemini-2.0-flash-001` (LLM)
- jokes.json (local frozen dataset)

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set up your .env**

```bash
cp .env.example .env
# add your OPENROUTER_API_KEY
```

**3. Start the app**

```bash
python app.py
```

Open http://localhost:5000

The jokes dataset is already included in `data/jokes.json`. If you want to re-fetch fresh jokes, run `python fetch.py` first.

## Project structure

```
app.py          - Flask app, loads jokes.json, handles /chat
fetch.py        - fetches 100 jokes from JokeAPI (run once)
data/jokes.json - frozen joke dataset
templates/      - chat UI
```
