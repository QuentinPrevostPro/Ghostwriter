import os
from dotenv import load_dotenv
from mistralai import Mistral
import json

load_dotenv()  # looks for .env in current dir / parents

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Parameters
input_file = "celine - voyage au bout de la nuit.txt"
output_file = "celine - voyage au bout de la nuit.jsonl"
chunk_size = 300  # number of words per chunk
model = "mistral-small-latest"

# Step 1: Load book and split into chunks
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()
chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 2: Generate pseudo-prompts using GPT
dataset = []

for idx, chunk in enumerate(chunks):
    print(f"Processing chunk {idx+1}/{len(chunks)}")

    # Generate a neutral/plain prompt
    response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that converts Céline-style text into neutral/plain text.",
            },
            {
                "role": "user",
                "content": f"Convert this Céline-style text into a neutral/plain version (1-2 sentences):\n\n{chunk}",
            },
        ],
        max_tokens=150,
    )
    pseudo_prompt = response.choices[0].message.content.strip()

    # Add to dataset
    dataset.append({
        "messages": [
            {
                "role": "user",
                "content": f"Neutral version of text: {pseudo_prompt} Rewrite in Céline style."
            },
            {
                "role": "assistant",
                "content": chunk
            }
        ]
    })

# Step 3: Save JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Dataset saved to {output_file}.")
