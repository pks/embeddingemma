"""Common utilities for embedding inference scripts."""

import torch

EXAMPLE_TEXTS = [
    "Where is the train station?",
    "Wo ist der Bahnhof?",
    "¿Dónde está la estación de tren?",
    "駅はどこですか？",
    "I like apples.",
]


def print_similarity_results(embeddings, texts):
    """Print embeddings shape and cosine distance matrix."""
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute cosine distance (embeddings are L2-normalized, so dist = 1 - dot product)
    sims = embeddings @ embeddings.T
    dists = 1 - sims
    print(f"\nCosine distance matrix:")
    for i, t in enumerate(texts):
        print(f"  {i}: {t[:40]}")
    torch.set_printoptions(sci_mode=False)
    print(dists)
