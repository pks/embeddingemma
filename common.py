"""Common utilities for embedding scripts."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

EXAMPLE_TEXTS = [
    "Where is the train station?",
    "Wo ist der Bahnhof?",
    "¿Dónde está la estación de tren?",
    "駅はどこですか？",
    "I like apples.",
]


class Embedder(nn.Module):
    def __init__(self, base_model, out_dim=768, layer=-1):
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


def load_tokenizer(model_id):
    """Load tokenizer with pad token set."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, legacy=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_id, device="cuda"):
    """Load frozen base model."""
    base = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    return base


def load_embedder(checkpoint_path, model_id, out_dim=768, layer=-1, device="cuda"):
    """Load embedder with trained projection weights."""
    base = load_base_model(model_id, device)
    embedder = Embedder(base, out_dim=out_dim, layer=layer).to(dtype=torch.bfloat16, device=device)
    embedder.proj.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    embedder.eval()
    return embedder


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
