import sqlite3
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

    def list_structure_types(self, table_name):
        table = self.db.open_table(table_name)
        structure_types = table.to_pandas()["type"].dropna().unique().tolist()
        return structure_types
    
    def get_biography(self, author):
        table = self.db.open_table("biography")
        df = table.to_pandas()
        row = df[df["author"] == author]
        if row.empty:
            return None
        return row.iloc[0]["biography"]

    def get_structure(self, structure_type):
        table = self.db.open_table("structure")
        df = table.to_pandas()
        row = df[df["type"] == structure_type]
        if row.empty:
            return None
        return row.iloc[0]["description"], row.iloc[0]["rules"]
