from shutil import ExecError
from app.utils.text_processing import load_text
from app.models.embedding_model import EmbeddingModel
from app.db.vector_store import VectorStore

#Parameters for the structure table
#description = "sources/structure/letters - description.txt"
#type = "letters"
#rules = "sources/structure/letters - rules.txt"
#table_name = "structure"

#Parameters for the author table
author = "Molière"
birth = "1622"
death = "1673"
biography = "sources/biography/moliere - biography.txt"
table_name = "biography"

def main():
    print("Loading source documents...")
    loaded_biography = load_text(biography)

    print("Saving to vector database...")
    store = VectorStore("./app/db/ghostwriter_db")
    #records = [{"description": loaded_description, "type": type, "rules": loaded_rules}]
    records = [{"author": author, "birth": birth, "death": death, "biography": loaded_biography}]
    try:
        store.append_table(table_name, records)
    except Exception:
        store.create_table(table_name, records)

    print("Done. Vector DB ready.")

if __name__ == "__main__":
    main()
