import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import sys


model_id = "MaLA-LM/emma-500-llama2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, legacy=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


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


def load_embedder(checkpoint_path="embedder.pt", device="cuda"):
    base = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    embedder = Embedder(base, out_dim=768, layer=-4).to(dtype=torch.bfloat16, device=device)
    embedder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    embedder.eval()
    return embedder


def embed(embedder, texts, max_length=512, device="cuda"):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        return embedder(**batch)


if __name__ == "__main__":
    embedder = load_embedder(sys.argv[1])

    # Example usage
    texts = [
        "Where is the train station?",
        "Wo ist der Bahnhof?",
        "¿Dónde está la estación de tren?",
        "駅はどこですか？",
        "I like apples.",
    ]

    embeddings = embed(embedder, texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute similarities
    sims = embeddings @ embeddings.T
    print(f"\nSimilarity matrix:")
    for i, t in enumerate(texts):
        print(f"  {i}: {t[:40]}")
    print(sims)
