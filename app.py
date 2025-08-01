from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import openai
import os
import json
import numpy as np

app = Flask(__name__)
CORS(app)

openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route("/", methods=["GET"])
def index():
    return "Bot backend is running.", 200


@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Load everything on demand (minimize RAM)
        index = faiss.read_index("faiss_index.pkl")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # very small model
        query_vec = model.encode([query])

        D, I = index.search(np.array(query_vec), k=5)
        context = "\n".join([chunks[i]["text"] for i in I[0]])

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant based on the published work of Kevin Spencer McCarthy. Only answer using the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )

        return jsonify({"response": completion["choices"][0]["message"]["content"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


