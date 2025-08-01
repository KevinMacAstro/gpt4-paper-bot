from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import openai

app = Flask(__name__)
CORS(app)

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load heavy stuff just once
model = None
index = None
chunks = None

def load_resources():
    global model, index, chunks
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    if index is None:
        index = faiss.read_index("faiss_index.pkl")
    if chunks is None:
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend is running."})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    load_resources()

    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=5)
    context = "\n".join([chunks[i]["text"] for i in I[0]])

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant based on the published research of Avenue McCarthy. Only use the context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    return jsonify({"response": response["choices"][0]["message"]["content"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)





