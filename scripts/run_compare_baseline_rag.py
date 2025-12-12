from app.comparison_utils.compare_baseline_rag import compare_baseline_rag

#Parameters
query = "Il fait beau dehors. Le soleil brille"
vector_db_path = "./app/db/ghostwriter_db"
embedding_model_name = "BAAI/bge-m3"
author = "Molière"
structure_type = "theater"
top_k = 5

def main():
    verdict = compare_baseline_rag(query, vector_db_path, embedding_model_name, author, structure_type, top_k)
    print(verdict)

if __name__ == "__main__":
    main()