# search_faq.py

import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load FAISS index
INDEX_DIR = "faiss_index"
index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))

# Load metadata (question-answer mapping)
with open(os.path.join(INDEX_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Function to get embeddings from OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # Smaller and cheaper; can use text-embedding-3-large for better accuracy
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


# Function to search FAISS index
def search_faq(query, top_k=3):
    query_vector = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
     
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            item = metadata[idx]
            result = {
                "question": item["question"],
                "answer": item["answer"],
                "score": float(dist)
            }
            # Include image_url if available
            if "image_url" in item:
                result["image_url"] = item["image_url"]
            results.append(result)
    return results

# If run directly, allow interactive search
if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        results = search_faq(user_query, top_k=1)
        if results:
            print("\nAnswer:", results[0]["answer"])
        else:
            print("\nNo match found.")
