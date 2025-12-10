from fastapi import FastAPI
from pydantic import BaseModel
from app.models.rag_model import RAGModel 

app = FastAPI()

# Instantiate your RAG model once (recommended)
rag = RAGModel("./app/db/ghostwriter_db", "BAAI/bge-m3")

# Request body model
class GenerateRequest(BaseModel):
    query: str
    author: str
    top_k: int = 5
    table_name: str = "content"


@app.post("/generate")
async def generate_response(request: GenerateRequest):
    result = rag.query(request.query, request.table_name, request.author, request.top_k,)
    return {"response": result}