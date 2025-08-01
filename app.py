from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Load data
index = faiss.read_index("faiss_index.pkl")
with open("chunks.json", "r") as f:
    chunks = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
openai.api_key = "YOUR_OPENAI_API_KEY"  # ← Replace this or load from env

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=5)
    context = "\n".join([chunks[i]["text"] for i in I[0]])

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant based on the published research of Avenue McCarthy. Only use the context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    return jsonify({"response": completion["choices"][0]["message"]["content"]})

if __name__ == "__main__":
    app.run(debug=True)


