#!/usr/bin/env python3
"""Run inference with trained embedder or sentence-transformer baseline."""

import argparse
import torch

from common import EXAMPLE_TEXTS, load_tokenizer, load_embedder, print_similarity_results


def embed(embedder, texts, tokenizer, max_length=512, device="cuda"):
    """Embed texts using the embedder."""
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        return embedder(**batch)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with trained embedder or baseline")

    # Checkpoint (optional - if not provided, uses sentence-transformer baseline)
    parser.add_argument("checkpoint", type=str, nargs="?", default=None,
                        help="Path to embedder checkpoint (.pt file). If not provided, uses --baseline model.")

    # Model
    parser.add_argument("--model", type=str, default="MaLA-LM/emma-500-llama3-8b-bi",
                        help="Base model ID (for checkpoint mode)")
    parser.add_argument("--baseline", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        help="Sentence-transformer model (when no checkpoint provided)")
    parser.add_argument("--out-dim", type=int, default=768,
                        help="Output embedding dimension")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Hidden layer to extract embeddings from")

    # Inference
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length")

    # Input
    parser.add_argument("--texts", type=str, nargs="+",
                        help="Texts to embed (if not provided, runs example)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    texts = args.texts if args.texts else EXAMPLE_TEXTS

    if args.checkpoint:
        # Use trained embedder
        print(f"Loading tokenizer from {args.model}...")
        tokenizer = load_tokenizer(args.model)

        print(f"Loading embedder from {args.checkpoint}...")
        embedder = load_embedder(
            args.checkpoint,
            model_id=args.model,
            out_dim=args.out_dim,
            layer=args.layer,
            device=args.device
        )

        embeddings = embed(embedder, texts, tokenizer, max_length=args.max_length, device=args.device)
    else:
        # Use sentence-transformer baseline
        from sentence_transformers import SentenceTransformer

        print(f"Loading baseline model {args.baseline}...")
        model = SentenceTransformer(args.baseline, device=args.device)

        embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    print_similarity_results(embeddings, texts)
