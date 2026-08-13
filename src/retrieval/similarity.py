import numpy as np

def dot_product(a, b):
    return float(np.dot(a, b))

def magnitude(vector):
    return float(np.linalg.norm(vector))

def cosine_similarity(a, b):
    norm_a = magnitude(a)
    norm_b = magnitude(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product(a, b) / (norm_a * norm_b)

