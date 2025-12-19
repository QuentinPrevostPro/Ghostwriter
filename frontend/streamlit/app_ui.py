import sys
from pathlib import Path

# Add project root (one level above frontend/) to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import requests
from app.db.vector_store import VectorStore    


st.title("Ghostwriter")

#Parameters
top_k = 5
table_name = "content"

# Input fields
query = st.text_area("Enter your text or idea")
store = VectorStore("./app/db/ghostwriter_db")
authors = store.list_authors("content")
author = st.selectbox("Select author", authors)
structure_types = store.list_structure_types("structure")
structure_type = st.selectbox("Select structure type", structure_types)


# Generate button
if st.button(f"Write like {author}"):
    if not query or not author:
        st.error("Please enter query, author and structure type")
    else:
        # Call the FastAPI endpoint
        try:
            response = requests.post(
                "http://127.0.0.1:8000/generate",
                json={
                    "query": query,
                    "author": author,
                    "structure_type": structure_type,
                    "top_k": top_k
                }
            )
            response.raise_for_status()
            data = response.json()
            st.text_area("Generated Response", value=data["response"], height=300)
        except Exception as e:
            st.error(f"Error calling API: {e}")
