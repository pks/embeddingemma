"""Run inference with sentence-transformer models."""

import argparse
import torch
from sentence_transformers import SentenceTransformer

from common import EXAMPLE_TEXTS, print_similarity_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with sentence-transformer models")

    # Model
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformer model ID")

    # Inference
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")

    # Input
    parser.add_argument("--texts", type=str, nargs="+",
                        help="Texts to embed (if not provided, runs example)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading model {args.model}...")
    model = SentenceTransformer(args.model, device=args.device)

    # Use provided texts or example
    texts = args.texts if args.texts else EXAMPLE_TEXTS

    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    print_similarity_results(embeddings, texts)
