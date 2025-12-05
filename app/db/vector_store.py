import lancedb

class VectorStore:
    def __init__(self, path):
        self.db = lancedb.connect(path)

    def create_table(self, table_name, records):
        return self.db.create_table(table_name, records)
    
    def similarity_search(self, query_embedding, table_name, top_k):
        table = self.db.open_table(table_name)
        results = table.search(query_embedding).limit(top_k).to_list()
        return results