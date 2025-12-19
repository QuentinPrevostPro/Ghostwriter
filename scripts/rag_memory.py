from app.models.rag_model import RAGModel
import time
from memory_profiler import memory_usage

def init_rag():
    rag = RAGModel("./app/db/ghostwriter_db", "BAAI/bge-m3")
    return rag

if __name__ == "__main__":
    print("Starting memory test...")
    mem_usage = memory_usage(init_rag)
    print("Memory usage during RAGModel initialization:", mem_usage)
    print("Test finished. Keep app alive 30s for monitoring...")
    time.sleep(30)
