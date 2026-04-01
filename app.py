import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from rag import rag_query

load_dotenv()

app = Flask(__name__)


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
        return jsonify({"error": "missing message"}), 400

    query = data["message"].strip()
    if not query:
        return jsonify({"error": "empty message"}), 400

    result = rag_query(query)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)