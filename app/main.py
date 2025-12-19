from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.models.rag_model import RAGModel
import sys

# Ensure stdout flushes immediately (useful for HF logs)
sys.stdout.reconfigure(line_buffering=True)

app = FastAPI()

# Enable CORS for Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAGModel at startup (no lazy loading)
print("⏳ Initializing RAGModel...")
rag = RAGModel("./app/db/ghostwriter_db", "BAAI/bge-m3")
print("✅ RAGModel initialized")

# Pydantic model for requests
class GenerateRequest(BaseModel):
    query: str
    author: str
    structure_type: str
    top_k: int = 5

# Health check endpoint
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    if rag:
        return {"status": "RAGModel loaded"}
    return {"status": "RAGModel not loaded"}

# Generate endpoint for Lovable
@app.post("/generate")
async def generate_response(request: GenerateRequest):
    print("➡️ /generate called")
    print("🧠 Running query...")
    result = rag.query(
        request.query,
        request.author,
        request.structure_type,
        request.top_k,
    )
    print("✅ Query finished")
    return {"response": result}