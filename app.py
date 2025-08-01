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

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Load model only when needed
        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        
        # Load index and chunks on demand
        index = faiss.read_index("faiss_index.pkl")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

        # Embed query and search
        query_vec = model.encode([query])
        D, I = index.search(np.array(query_vec), k=5)
        context = "\n".join([chunks[i]["text"][:500] for i in I[0]])  # Truncate to reduce token count

        # Call OpenAI
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant based on Avenue McCarthy's published research. Only use the context provided."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        )

        return jsonify({"response": completion.choices[0].message.content})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Something went wrong."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


