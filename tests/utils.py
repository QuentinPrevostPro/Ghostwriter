import os
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def judge_rag_score(output, author, nb_judges):
    scores = []
    for _ in range(nb_judges):
        prompt = (
            "Rate from 1 to 10 how much the following text resembles "
            f"{author}'s literary style. "
            "Only answer with the number.\n\n"
            f"{output}"
        )
        response = client.chat.complete(
            model="mistral-small-latest",  # or another Mistral model
            messages=[
                {
                    "role": "system",
                    "content": f"You are a LLM-as-a-judge to evaluate how close the generation is to {author}'s style. You return an integer score between 1 and 10."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        scores.append(int(response.choices[0].message.content))
    return sum(scores) / nb_judges