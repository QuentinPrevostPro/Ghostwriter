import os
from dotenv import load_dotenv
from mistralai import Mistral
from app.models.rag_model import RAGModel

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def generate_baseline(query, author): 
    prompt = f"Question:\n{query}\nAnswer in {author} style:"
    response = client.chat.complete(
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
        )
    return response.choices[0].message.content

def generate_rag(vector_db_path, embedding_model_name, query, author, structure_type, top_k): 
    rag = RAGModel(vector_db_path,embedding_model_name) #Initialize the RAG model
    return rag.query(query, author, structure_type, top_k) #Call the RAG workflow

def judge_compare_baseline_rag(baseline_output, rag_output, author):    
    prompt = (
        "Rate from 1 to 10 how much the two following texts resemble "
        f"{author}'s literary style. "
        "You must absolutely answer only with the grade for each text. It is forbidden to add labels\n\n"
        f"Baseline text : {baseline_output}\n\n"
        f"RAG text : {rag_output}"
    )
    response = client.chat.complete(
        model="mistral-small-latest",  # or another Mistral model
        messages=[
            {
                "role": "system",
                "content": f"You are a LLM-as-a-judge to evaluate which of the two input text is the closest to {author}'s style. You return an integer score between 1 and 10 for each text. Only an integer, labels are forbidden."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content
    

def compare_baseline_rag(query, vector_db_path, embedding_model_name, author, structure_type, top_k):
    baseline_output = generate_baseline(query, author)
    rag_output = generate_rag(vector_db_path, embedding_model_name, query, author, structure_type, top_k)
    scores = judge_compare_baseline_rag(baseline_output, rag_output, author)
    return scores

