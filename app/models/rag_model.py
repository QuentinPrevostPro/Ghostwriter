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

    def retrieve(self, query, author, structure_type, top_k): #Retrieve the top_k most relevant content chunks + author biography + text structure from the vector DB for a given query

        query_embedding = self.embedder.embed(query)
        text_chunks = self.store.similarity_search(query_embedding, "content", author, top_k)
        biography = self.store.get_biography(author)
        structure_description, structure_rules = self.store.get_structure(structure_type)
        return {
            "text_chunks": text_chunks,
            "biography": biography,
            "structure_description": structure_description,
            "structure_rules": structure_rules
        }

    def generate(self, query, context, author, structure_type): #Generate a response from the LLM using the query and retrieved context

        text = "\n\n".join([chunk["text"] for chunk in context["text_chunks"]])
        biography = context["biography"]
        structure_description = context["structure_description"]
        structure_rules = context["structure_rules"]

        prompt = f"""
            Writing structure description : 
            {structure_description}
            
            Writing structure rules : 
            {structure_rules}

            Author's biography : 
            {biography}

            Text:
            {text}
            
            Query: 
            {query}

            Answer in {author} style. The above text section is a source of inspiration for all stylistic effets. 
            The biography section gives you element about the life of the author that can influence the content of your ouput
            Writing strucure and rules must be respected
            """
                       
        response = self.llm.chat.complete(
            model="mistral-small-latest",  # or another Mistral model
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that writes in {author}'s style and that respects selected {structure_type}"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        return response.choices[0].message.content

    def query(self, query, author, structure_type, top_k): #Full RAG workflow: retrieve relevant context and generate the final response
        context = self.retrieve(query, author, structure_type, top_k)
        return self.generate(query, context, author, structure_type)