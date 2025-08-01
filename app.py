from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
CORS(app)

# Lazy-loaded globals
index = None
chunks = None
model = None

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with os.getenv(...) in production

def get_index():
    global index
    if index is None:
        index = faiss.read_index("faiss_index.pkl")
    return index

def get_chunks():
    global chunks
    if chunks is None:
        with open("chunks.json", "r") as f:
            chunks = json.load(f)
    return chunks

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Lazy load everything only when needed
    idx = get_index()
    ch = get_chunks()
    embedder = get_model()

    query_vec = embedder.encode([query])
    D, I = idx.search(np.array(query_vec), k=5)
    context = "\n".join([ch[i]["text"] for i in I[0]])

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant based on the published research of Kevin Spencer McCarthy. Only use the context provided."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        return jsonify({"response": completion["choices"][0]["message"]["content"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT env var
    app.run(host="0.0.0.0", port=port, debug=True)


