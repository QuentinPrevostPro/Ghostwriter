import time
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
        retrieve_start = time.time()
        print("  📊 Starting retrieval...")

        # Time embedding
        embed_start = time.time()
        query_embedding = self.embedder.embed(query)
        embed_time = time.time() - embed_start

        #Time vector search
        search_start = time.time()
        text_chunks = self.store.similarity_search(query_embedding, "content", author, top_k)
        search_time = time.time() - search_start
        print(f"  ⏱️  Vector search: {search_time:.3f}s")

        #Time biography / structure retrieval
        bio_start = time.time()
        biography = self.store.get_biography(author)
        structure_description, structure_rules = self.store.get_structure(structure_type)
        bio_time = time.time() - bio_start
        print(f"  ⏱️  Biography/structure: {bio_time:.3f}s")

        retrieve_time = time.time() - retrieve_start
        print(f"⏱️  Retrieve total: {retrieve_time:.2f}s")

        return {
            "text_chunks": text_chunks,
            "biography": biography,
            "structure_description": structure_description,
            "structure_rules": structure_rules
        }

    def generate(self, query, context, author, structure_type): #Generate a response from the LLM using the query and retrieved context
        generate_start = time.time()
        print("  ✍️  Starting generation...")

        #Time prompt construction
        prompt_start = time.time()
        text = "\n\n".join([chunk["text"] for chunk in context["text_chunks"]])
        biography = context["biography"]
        structure_description = context["structure_description"]
        structure_rules = context["structure_rules"]

        # Calculate target output length based on query
        query_length = len(query)
        print(f"  📐 Query length: {query_length} chars")

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

            IMPORTANT : Answer in {author} style. The above text section is a source of inspiration for all stylistic effets. 
            The biography section gives you element about the life of the author that can influence the content of your ouput (themes, perspectives or ideas)
            Writing structure and rules must be respected
            Full output must be written in the language of {author}
            Size of the be similar to the source text
            
            CRITICAL LENGTH CONSTRAINT : Your response must be approximately the same length as the query above. 
            The query is {query_length} characters long. Your response should be between {int(query_length * 0.8)} and {int(query_length * 1.2)} characters.
            Do not exceed this length limit."""
        
        prompt_time = time.time() - prompt_start
        prompt_size = len(prompt)
        estimated_tokens = prompt_size // 4
        print(f"  ⏱️  Prompt building: {prompt_time:.3f}s")
        print(f"  📏 Prompt size: {prompt_size:,} chars (~{estimated_tokens:,} tokens)")


        #Time LLM API call 
        print("  📡 Calling LLM API...")
        llm_start = time.time()          
        try:
            response = self.llm.chat.complete(
                model="mistral-small-latest",
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
                timeout_ms=120000,  # Add 120 second timeout
            )
            llm_time = time.time() - llm_start
            print(f"  ⏱️  LLM API call: {llm_time:.3f}s")
        except Exception as e:
            llm_time = time.time() - llm_start
            print(f"  ❌ LLM API call failed after {llm_time:.3f}s: {type(e).__name__}: {str(e)}")
            raise
        
        generate_time = time.time() - generate_start
        print(f"⏱️  Generate total: {generate_time:.2f}s")

        return response.choices[0].message.content

    def query(self, query, author, structure_type, top_k): #Full RAG workflow: retrieve relevant context and generate the final response
        context = self.retrieve(query, author, structure_type, top_k)
        return self.generate(query, context, author, structure_type)