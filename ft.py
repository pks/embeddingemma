import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


model_id = "MaLA-LM/emma-500-llama2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, legacy=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
base.eval()  # you may keep it in train() if using dropout; for embeddings often OK either way

# Option 1: freeze the LLM
for p in base.parameters():
    p.requires_grad = False

class Embedder(nn.Module):
    def __init__(self, base_model, out_dim=1024, layer=-4):
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
        h = out.hidden_states[self.layer]                 # [B,T,H]
        mask = attention_mask.unsqueeze(-1)               # [B,T,1]
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean pool
        z = self.proj(pooled)
        z = F.normalize(z, p=2, dim=1)

        return z

embedder = Embedder(base, out_dim=768, layer=-4).to(dtype=torch.bfloat16, device="cuda").train()
opt = torch.optim.AdamW(embedder.parameters(), lr=2e-4)

def tokenize(texts, max_length=128):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    return {k: v.cuda() for k, v in batch.items()}

def contrastive_loss(a, b, temp=0.05): # a,b are normalized [N,D]
    logits = (a @ b.T) / temp          # [N,N]
    labels = torch.arange(a.size(0), device=a.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_j) / 2

dataset = load_dataset(
    "MaLA-LM/FineOPUS-ReLID",
    data_dir="deu_Latn-eng_Latn",
    split="train",
    streaming=True,
)
dataset = dataset.shuffle(seed=42, buffer_size=10000)

batch_size = 32
val_size = 256

# Create fixed validation set
data_iter = iter(dataset)
val_left, val_right = [], []
for _ in range(val_size):
    sample = next(data_iter)
    val_left.append(sample["source_text"])
    val_right.append(sample["target_text"])
val_batch_l = tokenize(val_left)
val_batch_r = tokenize(val_right)

for step in range(1000):
    # load batch from dataset
    left, right = [], []
    for _ in range(batch_size):
        sample = next(data_iter)
        left.append(sample["source_text"])
        right.append(sample["target_text"])

    batch_l = tokenize(left)
    batch_r = tokenize(right)

    a = embedder(**batch_l)
    b = embedder(**batch_r)

    loss = contrastive_loss(a, b)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        # Evaluate on validation set
        embedder.eval()
        with torch.no_grad():
            val_a = embedder(**val_batch_l)
            val_b = embedder(**val_batch_r)
            val_loss = contrastive_loss(val_a, val_b)
        embedder.train()
        print(f"step {step} | train {loss.detach().item():.4f} | val {val_loss.item():.4f}")
