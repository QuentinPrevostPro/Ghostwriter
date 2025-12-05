import lancedb

class VectorStore:
    def __init__(self, path):
        self.db = lancedb.connect(path)

    def create_table(self, name, records):
        return self.db.create_table(name, records)