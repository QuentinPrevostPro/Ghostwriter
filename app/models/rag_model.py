import os
from dotenv import load_dotenv
from app.db.vector_store import VectorStore
from app.models.embedding_model import EmbeddingModel
from mistralai import Mistral

load_dotenv()

class RAGModel:
    def __init__(self, vector_db_path, embedding_model_name):
        self.store = VectorStore(vector_db_path)
        self.embedder = EmbeddingModel(embedding_model_name)
        self.llm = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def retrieve(self, query, table_name, author, top_k): #Retrieve the top_k most relevant chunks from the vector DB for a given query

        query_embedding = self.embedder.embed(query)
        results = self.store.similarity_search(query_embedding, table_name, author, top_k)
        return results

    def generate(self, query, context_chunks, author): #Generate a response from the LLM using the query and retrieved context
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
        prompt = f"Context:\n{context_text}\n\nQuestion:\n{query}\nAnswer in {author} style:"
        response = self.llm.chat.complete(
            model="mistral-small-latest",  # or another Mistral model
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that writes in {author}'s style."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content

    def query(self, query, table_name, author, top_k): #Full RAG workflow: retrieve relevant context and generate the final response
        chunks = self.retrieve(query, table_name, author, top_k)
        return self.generate(query, chunks, author)
