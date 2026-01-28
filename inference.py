#!/usr/bin/env python3
"""Run inference with trained embedder or sentence-transformer baseline."""

import sys
import os

# Check for --no-progress early, before imports
if "--no-progress" in sys.argv:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"

import argparse
import torch
from transformers import logging as hf_logging

from common import EXAMPLE_TEXTS, load_tokenizer, load_embedder, print_similarity_results, POOLING_MODES


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
    parser.add_argument("--pooling", type=str, default="mean", choices=POOLING_MODES,
                        help="Pooling strategy: mean, last, or attention")
    parser.add_argument("--mlp-head", action="store_true",
                        help="Use MLP projection head (must match training)")
    parser.add_argument("--mlp-hidden", type=int, default=2048,
                        help="Hidden dimension for MLP head (must match training)")

    # Inference
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length")

    # Input
    parser.add_argument("--texts", type=str, nargs="+",
                        help="Texts to embed (if not provided, runs example)")

    # Output
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Disable progress bars if requested (env vars already set at top of file)
    if args.no_progress:
        hf_logging.set_verbosity_error()

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
            device=args.device,
            pooling=args.pooling,
            mlp_head=args.mlp_head,
            mlp_hidden=args.mlp_hidden
        )

        embeddings = embed(embedder, texts, tokenizer, max_length=args.max_length, device=args.device)
    else:
        # Use sentence-transformer baseline
        from sentence_transformers import SentenceTransformer

        print(f"Loading baseline model {args.baseline}...")
        model = SentenceTransformer(args.baseline, device=args.device)

        embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    print_similarity_results(embeddings, texts)

    # Summary check for EXAMPLE_TEXTS
    if not args.texts and len(texts) == 5:
        import torch.nn.functional as F

        # Normalize embeddings for cosine similarity (float32 for precision)
        normed = F.normalize(embeddings.float(), dim=1)
        sim_matrix = normed @ normed.T

        # Sentences 0-3: same meaning (different languages)
        # Sentence 4: different meaning
        # Distance = 1 - cosine_similarity
        same_meaning_dists = []
        for i in range(4):
            for j in range(i + 1, 4):
                same_meaning_dists.append(1 - sim_matrix[i, j].item())
        avg_same_dist = sum(same_meaning_dists) / len(same_meaning_dists)

        # Distance of sentence 4 to sentences 0-3
        diff_meaning_dists = [1 - sim_matrix[4, i].item() for i in range(4)]
        avg_diff_dist = sum(diff_meaning_dists) / len(diff_meaning_dists)

        # Diagonal: self-distance should be 0
        diagonal = [1 - sim_matrix[i, i].item() for i in range(len(texts))]
        max_diag = max(abs(d) for d in diagonal)

        # Gap: different-meaning should be farther than same-meaning
        gap = avg_diff_dist - avg_same_dist

        print("\n" + "=" * 50)
        print("EXAMPLE_TEXTS Summary:")
        print(f"  Avg distance (same meaning, sentences 1-4): {avg_same_dist:.3f}")
        print(f"  Avg distance (different meaning, sentence 5 vs others): {avg_diff_dist:.3f}")
        print(f"  Gap: {gap:.3f}")
        print(f"  Diagonal (self-distance, should be 0): max={max_diag:.6f}")
        print("=" * 50)
