import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(responses, expected_responses):
    response_vecs = np.array([text_to_vector(response) for response in responses])
    expected_vecs = np.array([text_to_vector(response) for response in expected_responses])
    return cosine_similarity(response_vecs, expected_vecs).mean()

def text_to_vector(text):
    return np.array([ord(c) for c in text])  # A simple representation of text as numbers 