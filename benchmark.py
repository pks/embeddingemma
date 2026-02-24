#!/usr/bin/env python3
"""Evaluate multilingual embeddings on Tatoeba bitext-mining benchmark."""

import sys
import os

# Check for --no-progress early, before imports
if "--no-progress" in sys.argv:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"

import argparse
import json
import torch
import torch.nn.functional as F
from transformers import logging as hf_logging

from common import load_tokenizer, load_embedder, POOLING_MODES


def embed_batched(model, texts, tokenizer, batch_size, max_length, device, is_st=False):
    """Embed texts in batches, return (N, D) tensor in float32."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if is_st:
            emb = model.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)
        else:
            batch = tokenizer(batch_texts, padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                emb = model(**batch)
        all_embs.append(emb.float().cpu())
    return torch.cat(all_embs, dim=0)


def compute_metrics(en_embs, other_embs):
    """Compute retrieval metrics between English and other-language embeddings.

    Args:
        en_embs: (N, D) normalized float32 embeddings for English sentences
        other_embs: (N, D) normalized float32 embeddings for other-language sentences

    Returns:
        dict with P@1, P@5, MRR for both directions, and Gap.
        Pair i in en_embs corresponds to pair i in other_embs.
    """
    # Normalize for cosine similarity
    en_normed = F.normalize(en_embs, dim=1)
    other_normed = F.normalize(other_embs, dim=1)

    # Similarity matrix: (N, N)
    sim = en_normed @ other_normed.T  # sim[i,j] = similarity of en_i to other_j

    n = sim.size(0)
    targets = torch.arange(n)

    results = {}

    # en->X: for each English sentence, find its translation in other language
    # Row i of sim: similarities of en_i to all other_j. Correct answer is j=i.
    en_to_x_ranks = _compute_ranks(sim, targets)
    results["en_to_x_p@1"] = (en_to_x_ranks == 0).float().mean().item()
    results["en_to_x_p@5"] = (en_to_x_ranks < 5).float().mean().item()
    results["en_to_x_mrr"] = (1.0 / (en_to_x_ranks.float() + 1)).mean().item()

    # X->en: for each other-language sentence, find its translation in English
    # Column j of sim: similarities of all en_i to other_j. Correct answer is i=j.
    # Equivalent to transposing sim.
    x_to_en_ranks = _compute_ranks(sim.T, targets)
    results["x_to_en_p@1"] = (x_to_en_ranks == 0).float().mean().item()
    results["x_to_en_p@5"] = (x_to_en_ranks < 5).float().mean().item()
    results["x_to_en_mrr"] = (1.0 / (x_to_en_ranks.float() + 1)).mean().item()

    # Averaged metrics
    results["avg_p@1"] = (results["en_to_x_p@1"] + results["x_to_en_p@1"]) / 2
    results["avg_p@5"] = (results["en_to_x_p@5"] + results["x_to_en_p@5"]) / 2
    results["avg_mrr"] = (results["en_to_x_mrr"] + results["x_to_en_mrr"]) / 2

    # Gap: mean(different-pair distance) - mean(same-pair distance)
    dists = 1 - sim  # cosine distances
    same_pair_dists = dists.diag()  # distance between matching pairs
    # Off-diagonal: different-pair distances
    mask = ~torch.eye(n, dtype=torch.bool)
    diff_pair_dists = dists[mask]
    results["gap"] = diff_pair_dists.mean().item() - same_pair_dists.mean().item()

    return results


def _compute_ranks(sim_matrix, targets):
    """Compute rank of target for each row in similarity matrix.

    Args:
        sim_matrix: (N, M) similarities
        targets: (N,) index of correct match per row

    Returns:
        (N,) tensor of 0-indexed ranks
    """
    # Sort each row descending; find where the target lands
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)
    ranks = torch.zeros(sim_matrix.size(0), dtype=torch.long)
    for i in range(sim_matrix.size(0)):
        ranks[i] = (sorted_indices[i] == targets[i]).nonzero(as_tuple=True)[0][0]
    return ranks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multilingual embeddings on Tatoeba bitext-mining benchmark")

    # Checkpoint (optional - if not provided, uses sentence-transformer baseline)
    parser.add_argument("checkpoint", type=str, nargs="?", default=None,
                        help="Path to embedder checkpoint (.pt file)")

    # Model
    parser.add_argument("--model", type=str, default="MaLA-LM/emma-500-llama3-8b-bi",
                        help="Base model ID (for checkpoint mode)")
    parser.add_argument("--baseline", type=str,
                        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        help="Sentence-transformer model (when no checkpoint provided)")
    parser.add_argument("--out-dim", type=int, default=768,
                        help="Output embedding dimension")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Hidden layer to extract embeddings from")
    parser.add_argument("--pooling", type=str, default="mean", choices=POOLING_MODES,
                        help="Pooling strategy")
    parser.add_argument("--mlp-head", action="store_true",
                        help="Use MLP projection head (must match training)")
    parser.add_argument("--mlp-hidden", type=int, default=2048,
                        help="Hidden dimension for MLP head (must match training)")

    # Benchmark
    parser.add_argument("--benchmark-file", type=str, default="benchmark.json",
                        help="Path to benchmark data file")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for embedding")
    parser.add_argument("--languages", type=str, nargs="+", default=None,
                        help="Subset of languages to evaluate (e.g., de ja zh)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save structured JSON results")

    # Inference
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length")

    # Output
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_progress:
        hf_logging.set_verbosity_error()

    # Load benchmark data
    with open(args.benchmark_file, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Filter languages
    lang_codes = list(benchmark["pairs"].keys())
    if args.languages:
        lang_codes = [l for l in args.languages if l in benchmark["pairs"]]
        missing = set(args.languages) - set(lang_codes)
        if missing:
            print(f"Warning: languages not in benchmark: {missing}")

    print(f"Evaluating {len(lang_codes)} languages: {', '.join(lang_codes)}")

    # Load model
    tokenizer = None
    is_st = False

    if args.checkpoint:
        print(f"Loading tokenizer from {args.model}...")
        tokenizer = load_tokenizer(args.model)
        print(f"Loading embedder from {args.checkpoint}...")
        model = load_embedder(
            args.checkpoint,
            model_id=args.model,
            out_dim=args.out_dim,
            layer=args.layer,
            device=args.device,
            pooling=args.pooling,
            mlp_head=args.mlp_head,
            mlp_hidden=args.mlp_hidden,
        )
        model_name = os.path.basename(args.checkpoint)
    else:
        from sentence_transformers import SentenceTransformer
        print(f"Loading baseline model {args.baseline}...")
        model = SentenceTransformer(args.baseline, device=args.device)
        is_st = True
        model_name = args.baseline

    # Evaluate each language
    all_results = {}

    for code in lang_codes:
        pairs = benchmark["pairs"][code]
        lang_name = benchmark["languages"][code]["name"]
        n = len(pairs)

        en_texts = [p["en"] for p in pairs]
        other_texts = [p["other"] for p in pairs]

        print(f"\n{lang_name} ({code}): {n} pairs")
        print(f"  Embedding English...", end="", flush=True)
        en_embs = embed_batched(model, en_texts, tokenizer, args.batch_size,
                                args.max_length, args.device, is_st)
        print(f" done ({en_embs.shape})")

        print(f"  Embedding {lang_name}...", end="", flush=True)
        other_embs = embed_batched(model, other_texts, tokenizer, args.batch_size,
                                   args.max_length, args.device, is_st)
        print(f" done ({other_embs.shape})")

        metrics = compute_metrics(en_embs, other_embs)
        all_results[code] = metrics

    # Print formatted table
    print("\n" + "=" * 90)
    print(f"Benchmark Results: {model_name}")
    print("=" * 90)
    header = f"{'Lang':<6} {'en→X P@1':>9} {'X→en P@1':>9} {'Avg P@1':>8} {'Avg P@5':>8} {'Avg MRR':>8} {'Gap':>7}"
    print(header)
    print("-" * 90)

    # Per-language rows
    avg_metrics = {k: 0.0 for k in ["avg_p@1", "avg_p@5", "avg_mrr", "gap",
                                      "en_to_x_p@1", "x_to_en_p@1"]}
    for code in lang_codes:
        m = all_results[code]
        print(f"{code:<6} {m['en_to_x_p@1']:>9.1%} {m['x_to_en_p@1']:>9.1%} "
              f"{m['avg_p@1']:>8.1%} {m['avg_p@5']:>8.1%} {m['avg_mrr']:>8.3f} "
              f"{m['gap']:>7.3f}")
        for k in avg_metrics:
            avg_metrics[k] += m[k]

    # Average row
    n_langs = len(lang_codes)
    for k in avg_metrics:
        avg_metrics[k] /= n_langs
    print("-" * 90)
    print(f"{'AVG':<6} {avg_metrics['en_to_x_p@1']:>9.1%} {avg_metrics['x_to_en_p@1']:>9.1%} "
          f"{avg_metrics['avg_p@1']:>8.1%} {avg_metrics['avg_p@5']:>8.1%} "
          f"{avg_metrics['avg_mrr']:>8.3f} {avg_metrics['gap']:>7.3f}")
    print("=" * 90)

    # Save JSON results
    if args.output_json:
        output = {
            "model": model_name,
            "checkpoint": args.checkpoint,
            "layer": args.layer,
            "pooling": args.pooling,
            "mlp_head": args.mlp_head,
            "per_language": all_results,
            "average": avg_metrics,
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
