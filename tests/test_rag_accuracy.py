import pytest
from app.models.rag_model import RAGModel
from tests.utils import judge_rag_score
from app.comparison_utils.compare_baseline_rag import compare_baseline_rag

#LLM-as-a-judge on the RAG model
def test_style_quality_minimum_score():
    """
    Ensures the RAG model produces a minimum acceptable author-like style.
    """
    rag = RAGModel("./app/db/ghostwriter_db","BAAI/bge-m3")
    output = rag.query("Il fait beau dehors. Le soleil brille", "Louis-Ferdinand Céline", "prose", 5)

    score = judge_rag_score(output,"Louis-Ferdinand Céline",3)

    # Minimum threshold — adjust as your model improves
    assert score >= 7, f"Style score too low: {score}"

#Baseline vs RAG comparison
def test_baseline_rag_comparison():
    """
    Ensures the RAG model outperforms the baseline model
    """
    scores = compare_baseline_rag("Il fait beau dehors. Le soleil brille", "./app/db/ghostwriter_db", "BAAI/bge-m3", "Louis-Ferdinand Céline", "prose", 5)
    baseline_score, rag_score = map(int, scores.split())


    #RAG score must always be higher than baseline score
    assert rag_score >= baseline_score, f"RAG model is less accurate than the baseline model\n\nBaseline score : {baseline_score}\n\nRAG score : {rag_score}\n\n"
