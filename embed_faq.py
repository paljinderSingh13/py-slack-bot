import os
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# 2. Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# 3. Load FAQ dataset
faq_file = "faq.json"
if not os.path.exists(faq_file):
    raise FileNotFoundError(f"{faq_file} not found.")

with open(faq_file, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# 4. Prepare questions for embedding
questions = [item["question"] for item in faq_data]

# 5. Create embeddings
print("Generating embeddings...")
embeddings = []
for q in questions:
    response = client.embeddings.create(
        input=q,
        model="text-embedding-3-small"  # Lower cost; can use text-embedding-3-large for more accuracy
    )
    embeddings.append(response.data[0].embedding)

embeddings_array = np.array(embeddings).astype("float32")

# 6. Create FAISS index
embedding_dim = len(embeddings_array[0])
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_array)

# 7. Save FAISS index & metadata
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/index.faiss")

with open("faiss_index/metadata.json", "w", encoding="utf-8") as f:
    json.dump(faq_data, f, ensure_ascii=False, indent=2)

print(f"✅ FAISS index saved in 'faiss_index/' with {len(questions)} questions.")
