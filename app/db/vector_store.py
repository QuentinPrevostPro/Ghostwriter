import lancedb
import pandas

class VectorStore:
    def __init__(self, path):
        self.db = lancedb.connect(path)

    def create_table(self, table_name, records):
        return self.db.create_table(table_name, records)
    
    def append_table(self, table_name, records):
        table = self.db.open_table(table_name)
        return table.add(records)
 
    def similarity_search(self, query_embedding, table_name, author, top_k):
        table = self.db.open_table(table_name)
        results = table.search(query_embedding).where(f"author == '{author}'").limit(top_k).to_list()
        return results
    
    def list_authors(self, table_name):
        table = self.db.open_table(table_name)
        authors = table.to_pandas()["author"].dropna().unique().tolist()
        return authors
    
