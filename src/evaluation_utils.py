import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def eval_single_response_translation(expected_answer: str, llm_response: str) -> float:
    """From colab notebook: Compares an LLM response to the expected response using cosine similarity."""
    if not expected_answer or not llm_response:
        return 0.0

    try:
        sentences = [expected_answer, llm_response]
        embeddings = model.encode(sentences)
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return float(similarity.item())
    except Exception as e:
        print(f"Error using sentence-transformers: {e}")
        pass

def eval_single_response_classification(expected_answer: str, llm_response: str) -> float:
    """From colab notebook: Compares an LLM response to the expected response."""
    if not expected_answer or not llm_response:
        return 0.0

    expected = expected_answer.lower().strip()
    response = llm_response.lower().strip()

    return float(expected == response)

def evaluate(query_fn, eval_fn, dataset) -> float:
    """From colab notebook: Computes aggregate score across the dataset."""
    total_score = 0.0
    total_items = 0

    for item in dataset:
        try:
            llm_response = query_fn(item["post"])
            score = eval_fn(item["expected_answer"], llm_response)
            total_score += score
            total_items += 1
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue

    return (total_score / total_items) * 100 if total_items > 0 else 0.0