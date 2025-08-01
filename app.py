from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import json
import os
import openai
import faiss
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")  # Serve your frontend

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not hasattr(app, "index"):
        app.index = faiss.read_index("faiss_index.pkl")
    if not hasattr(app, "chunks"):
        with open("chunks.json", "r") as f:
            app.chunks = json.load(f)
    if not hasattr(app, "model"):
        app.model = SentenceTransformer("all-MiniLM-L6-v2")

    query_vec = app.model.encode([query])
    D, I = app.index.search(np.array(query_vec), k=5)
    context = "\n".join([app.chunks[i]["text"] for i in I[0]])

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant based on the published research of Avenue McCarthy. Only use the context provided."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return jsonify({"response": completion["choices"][0]["message"]["content"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)



