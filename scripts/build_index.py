import json
import numpy as np
from utils.embeddings import hf_embedding
from utils.indexing import build_faiss_index

with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# texts = [c["text"] for c in chunks]
# embeddings = np.vstack([hf_embedding(t) for t in texts])

# np.save("data/embeddings.npy", embeddings)

# index = build_faiss_index(embeddings)
# save_faiss(index, "data/faiss.index")

build_faiss_index(chunks, hf_embedding, index_path="data/faiss.index", embeddings_path="data/embeddings.npy")
# print(f"Indexed {len(texts)} chunks")

