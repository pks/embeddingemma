import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

model_id = "MaLA-LM/emma-500-llama2-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # or float16
    device_map="auto"
)
model.eval()

# Llama tokenizers often have no pad_token; set it for batching.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@torch.no_grad()
def embed(texts, max_length=256, layer=-2, normalize=True):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {k: v.to(model.device) for k, v in batch.items()}

    out = model(**batch, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer]            # [B, T, H]
    mask = batch["attention_mask"].unsqueeze(-1)  # [B, T, 1]

    # mean pool over non-pad tokens
    emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    if normalize:
        emb = F.normalize(emb, p=2, dim=1)

    return emb  # torch tensor [B, H]

def cosine_sim(a, b):
    # assumes normalized embeddings; then dot product == cosine
    return a @ b.T

sentences = ["A cat sits on the mat.", "A dog lies on the rug.", "Quantum mechanics is hard.", "Eine Katze sitzt auf der Matte."]
sentences = [f"Represent the meaning of the following text: {s}" for s in sentences]
layers = range(0, 32)
for layer in layers:
    E = embed(sentences, layer=layer)
    S = 1 - cosine_sim(E, E)
    print(layer, S)
    print()
