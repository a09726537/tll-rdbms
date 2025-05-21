import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# === CONFIG ===
KB_CSV_PATH = "data/kb_corpus.csv"         # Your corpus (text + optional metadata)
TEXT_COLUMN = "text"                       # The column in the CSV to embed
INDEX_OUTPUT = "retriever/bert_retriever.faiss"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 64
SEED = 42

# === LOAD DATA ===
print("Loading knowledge base corpus...")
df = pd.read_csv(KB_CSV_PATH)
texts = df[TEXT_COLUMN].astype(str).tolist()

# === EMBEDDING ===
print(f"Embedding {len(texts)} entries using {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)

# === NORMALIZATION (optional for cosine similarity) ===
embeddings = embeddings.astype("float32")
faiss.normalize_L2(embeddings)

# === BUILD FAISS INDEX ===
print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Use IndexFlatL2 for Euclidean
index.add(embeddings)

# === SAVE INDEX ===
os.makedirs(os.path.dirname(INDEX_OUTPUT), exist_ok=True)
faiss.write_index(index, INDEX_OUTPUT)
print(f"FAISS index saved to: {INDEX_OUTPUT}")
