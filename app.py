"""
app.py — Flask application
----------------------------
Serves the chat UI and exposes a /chat API endpoint.

Routes:
    GET  /          → Chat UI
    POST /chat      → RAG query, returns JSON
    GET  /health    → Health check
    POST /reset     → Clear conversation history
"""

import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from rag import rag_query

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session-based conversation history


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": os.getenv("MODEL", "google/gemini-2.0-flash-001")})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    query = data["message"].strip()
    if not query:
        return jsonify({"error": "Empty message"}), 400

    # Retrieve conversation history from session (multi-turn support)
    history = session.get("history", [])

    # Run RAG pipeline
    result = rag_query(query, history=history)

    # Append to history (store assistant answer for context in future turns)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": result["answer"]})

    # Keep history bounded to last 10 turns (20 messages) to avoid token bloat
    if len(history) > 20:
        history = history[-20:]

    session["history"] = history

    return jsonify({
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
    })


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("history", None)
    return jsonify({"status": "conversation reset"})


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Starting LLM Challenge app on http://localhost:{port}")
    print(f"   Model: {os.getenv('MODEL', 'google/gemini-2.0-flash-001')}")
    print(f"   Debug: {debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
