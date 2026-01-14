"""Run inference with trained embedder."""

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
    parser = argparse.ArgumentParser(description="Run inference with trained embedder")

    # Required
    parser.add_argument("checkpoint", type=str,
                        help="Path to embedder checkpoint (.pt file)")

    # Model
    parser.add_argument("--model", type=str, default="MaLA-LM/emma-500-llama3-8b-bi",
                        help="Base model ID")
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

    # Use provided texts or example
    texts = args.texts if args.texts else EXAMPLE_TEXTS

    embeddings = embed(embedder, texts, tokenizer, max_length=args.max_length, device=args.device)
    print_similarity_results(embeddings, texts)
