from app.db.vector_store import VectorStore
from app.models.rag_model import RAGModel

#Parameters
vector_db_path = "./app/db/ghostwriter_db"
embedding_model_name = "BAAI/bge-m3"
structure_type = "prose"
top_k = 5

def pick_author(store):
    authors = store.list_authors("content")
    if not authors:
        raise RuntimeError("No authors found in the table.")
    print("Available authors:")
    for i, a in enumerate(authors, 1):
        print(f"{i}. {a}")
    while True:
        choice = input("Select an author (number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(authors):
            return authors[int(choice) - 1]
        print("Invalid choice. Try again.")

def main():
    store = VectorStore(vector_db_path)
    rag = RAGModel(vector_db_path,embedding_model_name) #Initialize the RAG model
    print("=== Ghostwriter RAG CLI ===")
    author = pick_author(store)
    print(f"Selected author: {author}.\nType your query or idea (Ctrl+C to exit)\n")
    while True:
        query = input("Your query: ").strip()
        if not query:
            continue

        result = rag.query(query, author, structure_type, top_k) #Call the RAG workflow
        print("\n--- Response ---")
        print(result)
        print("\n====================\n")

if __name__ == "__main__":
    main()
