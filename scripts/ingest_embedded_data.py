from shutil import ExecError
from app.utils.text_processing import load_text, chunk_text
from app.models.embedding_model import EmbeddingModel
from app.db.vector_store import VectorStore

#Parameters for the content table
text = "sources/moliere - dom juan.txt"
type = "theater"
author = "Molière"
title = "Dom Juan ou le Festin de Pierre"
date = 1665
chunk_size = 300
overlap = 50
model = "BAAI/bge-m3"
table_name = "content"

def main():
    print("Loading source text...")
    loaded_text= load_text(text)

    print("Chunking text...")
    chunks = chunk_text(loaded_text, chunk_size, overlap)
    print(f"Total chunks created: {len(chunks)}")

    print("Loading embedding model...")
    embedder = EmbeddingModel(model)

    print("Embedding chunks...")
    embeddings = []
    for idx, chunk in enumerate(chunks):
        emb = embedder.embed(chunk)
        embeddings.append(emb)
        print(f"Embedded {idx}/{len(chunks)} chunks")


    print("Saving to vector database...")
    store = VectorStore("./app/db/ghostwriter_db")
    records = [{"text": chunk, "embedding": emb, "type": type, "author": author, "title": title, "date": date} for chunk, emb in zip(chunks, embeddings)]
    try:
        store.append_table(table_name, records)
    except Exception:
        store.create_table(table_name, records)

    print("Done. Vector DB ready.")

if __name__ == "__main__":
    main()
