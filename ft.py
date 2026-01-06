import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import threading
from tqdm import tqdm
from queue import Queue


model_id = "MaLA-LM/emma-500-llama2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, legacy=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
base.eval()
for p in base.parameters():
    p.requires_grad = False

torch.cuda.empty_cache()

val_base = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:1")
val_base.eval()
for p in val_base.parameters():
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

embedder = Embedder(base, out_dim=768, layer=-4).to(dtype=torch.bfloat16, device="cuda:0").train()
val_embedder = Embedder(val_base, out_dim=768, layer=-4).to(dtype=torch.bfloat16, device="cuda:1").eval()
opt = torch.optim.AdamW(embedder.parameters(), lr=2e-4)

def tokenize(texts, max_length=128, device="cuda:0"):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}

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

max_batch_tokens = 2048  # total tokens per batch (across all samples)
max_length = 128
val_size = 32

# Create fixed validation set
data_iter = iter(dataset)
val_left, val_right = [], []
for _ in range(val_size):
    sample = next(data_iter)
    val_left.append(sample["source_text"])
    val_right.append(sample["target_text"])
val_batch_l = tokenize(val_left, device="cuda:1")
val_batch_r = tokenize(val_right, device="cuda:1")

# Async validation
val_thread = None
val_loss_result = [None]  # use list for mutability in thread

def run_validation():
    # Copy projection weights to cuda:1
    state = {k: v.to("cuda:1") for k, v in embedder.proj.state_dict().items()}
    val_embedder.proj.load_state_dict(state)
    with torch.no_grad():
        val_a = val_embedder(**val_batch_l)
        val_b = val_embedder(**val_batch_r)
        loss = contrastive_loss(val_a, val_b)
    val_loss_result[0] = loss.item()

def estimate_tokens(text, max_len):
    """Fast token estimate: ~4 chars per token for LLaMA."""
    return min(len(text) // 4 + 1, max_len)

def get_batch(data_iter, max_tokens, max_len):
    """Accumulate samples until we hit the token budget."""
    left, right = [], []
    total_tokens = 0
    while total_tokens < max_tokens:
        sample = next(data_iter)
        src, tgt = sample["source_text"], sample["target_text"]
        total_tokens += max(estimate_tokens(src, max_len), estimate_tokens(tgt, max_len))
        left.append(src)
        right.append(tgt)
    return left, right

# Prefetch batches in background
batch_queue = Queue(maxsize=4)

def prefetch_worker():
    while True:
        left, right = get_batch(data_iter, max_batch_tokens, max_length)
        batch_l = tokenize(left, max_length=max_length, device="cuda:0")
        batch_r = tokenize(right, max_length=max_length, device="cuda:0")
        batch_queue.put((batch_l, batch_r))

prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
prefetch_thread.start()

pbar = tqdm(range(1000), desc="Training")
for step in pbar:
    batch_l, batch_r = batch_queue.get()

    a = embedder(**batch_l)
    b = embedder(**batch_r)

    loss = contrastive_loss(a, b)

    opt.zero_grad()
    loss.backward()
    opt.step()

    # Update progress bar with train loss
    pbar.set_postfix(train=f"{loss.detach().item():.4f}", val=f"{val_loss_result[0]:.4f}" if val_loss_result[0] else "...")

    if step % 100 == 0:
        # Wait for previous validation if running
        if val_thread is not None:
            val_thread.join()
        # Launch new validation on cuda:1
        val_thread = threading.Thread(target=run_validation)
        val_thread.start()
        # For step 0, wait for result
        if step == 0:
            val_thread.join()
            pbar.set_postfix(train=f"{loss.detach().item():.4f}", val=f"{val_loss_result[0]:.4f}")
        # Save checkpoint
        torch.save(embedder.state_dict(), f"embedder_step{step}.pt")

# Wait for final validation
if val_thread is not None:
    val_thread.join()

# Save the model
torch.save(embedder.state_dict(), "embedder.pt")
print(f"Model saved to embedder.pt")
