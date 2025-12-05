from app.models.rag_model import RAGModel

#Parameters
vector_db_path = "./app/db/ghostwriter_db"
embedding_model_name = "BAAI/bge-m3"
table_name = "celine"


def main():
    rag = RAGModel(vector_db_path,embedding_model_name) #Initialize the RAG model
    print("=== Ghostwriter RAG CLI ===")
    print("Type your query or idea (Ctrl+C to exit)\n")
    while True:
        query = input("Your query: ").strip()
        if not query:
            continue

        result = rag.query(query, table_name, top_k=5) #Call the RAG workflow
        print("\n--- Response ---")
        print(result)
        print("\n====================\n")

if __name__ == "__main__":
    main()
