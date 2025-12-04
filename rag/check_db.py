import lancedb
db = lancedb.connect("./ghostwriter_db")

print(db.table_names())

table = db["celine"]
print(table.schema)

rows = table.to_pandas().head(5)
print(rows)
print("Number of records:", len(table.to_pandas()))

