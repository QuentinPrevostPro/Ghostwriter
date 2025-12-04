import os
import time
import lancedb
from sentence_transformers import SentenceTransformer

# Parameters
input_file = "../sources/celine - voyage au bout de la nuit.txt"
chunk_size = 300 #300 words per chunk
model = SentenceTransformer("BAAI/bge-m3")

# Split the text into chunks
def chunk_text(input_file, chunk_size, overlap=50):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    chunks = []
    start = 0
        
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move with overlap    return chunks

    return chunks

# Embed the chunks

def embed_chunks(chunks):
    embeddings = []
    for idx, chunk in enumerate(chunks):
        emb = model.encode(chunk)
        embeddings.append(emb)
        print(f"{idx+1}/{len(chunks)}")
    return embeddings

#Store embeddings in a vector db
chunks = chunk_text(input_file, chunk_size)
embeddings = embed_chunks(chunks)

records = [{"text":  chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
db = lancedb.connect("./ghostwriter_db")
table = db.create_table("celine", data=records)

print("Vector DB ready.")


