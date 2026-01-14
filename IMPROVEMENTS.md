# Cross-lingual Embedding Improvements

## Problem: Tokenization Disparity

Observed token counts per sentence vary significantly across languages:
- zh/ar: ~8 tokens (very efficient)
- en: ~14 tokens
- kr: ~22 tokens
- de: ~34 tokens
- gu/pa: ~38 tokens

This 4-5x difference affects mean pooling: each token contributes differently to the pooled representation depending on language.

---

## Implemented Options

### Pooling (`--pooling`)

1. ✅ **Mean pooling** (default)
   - `--pooling mean`
   - Original behavior, averages all tokens

2. ✅ **Last token pooling**
   - `--pooling last`
   - Uses final non-padded token (LLM-style)
   - Works well for decoder models

3. ✅ **Attention pooling**
   - `--pooling attention`
   - Learnable query vector over tokens
   - Directly addresses length disparity
   - Adds ~4K parameters

### Projection Head (`--mlp-head`)

4. ✅ **MLP head**
   - `--mlp-head`
   - Linear(4096→2048) → ReLU → Linear(2048→768)
   - Configurable with `--mlp-hidden N`
   - ~18M params (default) vs ~3M for linear

---

## Options to Explore

### Pooling

5. **Length-normalized mean pooling**
   - Scale pooled output by sqrt(length) or similar factor
   - Simple change, might help balance representations

6. **Max pooling**
   - Take max per dimension instead of mean
   - Less sensitive to sequence length

7. **Weighted combination of pooling strategies**
   - e.g., 0.5 * mean + 0.5 * max

### Projection Head

8. **Per-language-family heads**
   - Separate projections for CJK, Latin, Cyrillic, Arabic, Indic
   - Requires language detection/routing

9. **Mixture-of-experts head**
   - Soft routing based on input representation
   - More flexible than hard language families

10. **Deeper head with LayerNorm + residual**
    - Better gradient flow
    - More stable training

### LoRA on Base Model (currently frozen)

11. **LoRA on attention layers (q, v projections)**
    - Adapts representations themselves, not just projection
    - Typical rank: 8-64

12. **Language-family-specific LoRA adapters**
    - Different LoRA weights for different scripts
    - Most flexible but complex

13. **LoRA on last N layers only**
    - e.g., last 4 layers
    - Reduces parameter count, focuses on high-level features

### Combined Approaches

14. **LoRA + attention pooling**
    - Both representation and pooling are learnable

15. **LoRA + per-family heads**
    - Adapt base model + specialized projections

---

## Parameter Counts

| Configuration | Params |
|--------------|--------|
| Linear projection only | ~3M |
| + Attention pooling | +4K |
| + MLP head (2048) | ~18M |
| + MLP head (4096) | ~35M |
| + LoRA (r=16, last 4 layers) | +10-20M |

---

## Usage Examples

```bash
# Baseline (mean pooling, linear head)
./finetuning.py

# Last token pooling
./finetuning.py --pooling last

# Attention pooling
./finetuning.py --pooling attention

# MLP head
./finetuning.py --mlp-head

# MLP head with custom hidden dim
./finetuning.py --mlp-head --mlp-hidden 1024

# Combined: attention pooling + MLP head
./finetuning.py --pooling attention --mlp-head
```

---

## Alternative Loss Functions

For reference, alternatives to current contrastive/InfoNCE loss:

- **Triplet loss**: anchor/positive/negative with margin
- **Multiple Negatives Ranking Loss (MNRL)**: sentence-transformers default
- **Hard negative mining**: mine similar but non-parallel pairs
- **Circle loss**: adaptive margins based on similarity
- **Knowledge distillation**: align to teacher model
