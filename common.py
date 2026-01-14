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

POOLING_MODES = ["mean", "last", "attention", "max"]


class Embedder(nn.Module):
    def __init__(self, base_model, out_dim=768, layer=-1, pooling="mean", mlp_head=False, mlp_hidden=None):
        super().__init__()
        self.base = base_model
        self.layer = layer
        self.pooling = pooling
        self.mlp_head = mlp_head
        hidden = base_model.config.hidden_size

        # Attention pooling: learnable query vector
        if pooling == "attention":
            self.attn_query = nn.Parameter(torch.randn(hidden))

        # Projection head
        if mlp_head:
            mlp_dim = mlp_hidden if mlp_hidden is not None else hidden
            self.proj = nn.Sequential(
                nn.Linear(hidden, mlp_dim, bias=False),
                nn.ReLU(),
                nn.Linear(mlp_dim, out_dim, bias=False),
            )
        else:
            self.proj = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[self.layer]  # (batch, seq, hidden)

        # Pooling
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        elif self.pooling == "last":
            # Get last non-padded token for each sequence
            seq_lens = attention_mask.sum(dim=1) - 1  # (batch,)
            pooled = h[torch.arange(h.size(0), device=h.device), seq_lens]
        elif self.pooling == "attention":
            # Attention pooling with learnable query
            scores = torch.matmul(h, self.attn_query)  # (batch, seq)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)
            pooled = (h * weights).sum(1)  # (batch, hidden)
        elif self.pooling == "max":
            # Max pooling (mask padded positions with -inf)
            mask = attention_mask.unsqueeze(-1)
            h_masked = h.masked_fill(mask == 0, float('-inf'))
            pooled = h_masked.max(dim=1).values  # (batch, hidden)

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
    base = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    return base


def load_embedder(checkpoint_path, model_id, out_dim=768, layer=-1, device="cuda",
                  pooling="mean", mlp_head=False, mlp_hidden=None):
    """Load embedder with trained projection weights."""
    base = load_base_model(model_id, device)
    embedder = Embedder(base, out_dim=out_dim, layer=layer, pooling=pooling,
                        mlp_head=mlp_head, mlp_hidden=mlp_hidden).to(dtype=torch.bfloat16, device=device)
    embedder.proj.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    # Load attention query if present
    if pooling == "attention":
        try:
            attn_state = torch.load(checkpoint_path.replace('.pt', '_attn.pt'),
                                    map_location=device, weights_only=True)
            embedder.attn_query.data = attn_state['attn_query']
        except FileNotFoundError:
            pass  # Use random init if no saved attention weights
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
