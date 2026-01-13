import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import argparse

from common import EXAMPLE_TEXTS, print_similarity_results


class Embedder(nn.Module):
    def __init__(self, base_model, out_dim=768, layer=-4):
        super().__init__()
        self.base = base_model
        self.layer = layer
        hidden = base_model.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[self.layer]
        mask = attention_mask.unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        z = self.proj(pooled)
        z = F.normalize(z, p=2, dim=1)
        return z


def load_embedder(checkpoint_path, model_id="MaLA-LM/emma-500-llama2-7b",
                  out_dim=768, layer=-4, device="cuda"):
    base = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    embedder = Embedder(base, out_dim=out_dim, layer=layer).to(dtype=torch.bfloat16, device=device)
    embedder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    embedder.eval()
    return embedder


def load_tokenizer(model_id="MaLA-LM/emma-500-llama2-7b"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, legacy=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def embed(embedder, texts, tokenizer, max_length=512, device="cuda"):
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
