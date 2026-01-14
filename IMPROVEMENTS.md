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

## Options to Explore

### Pooling (addressing length disparity)

1. **Length-normalized mean pooling**
   - Scale pooled output by sqrt(length) or similar factor
   - Simple change, might help balance representations

2. **Max pooling**
   - Take max per dimension instead of mean
   - Less sensitive to sequence length

3. **Attention pooling** (learnable query vector)
   - Learn which tokens matter for the embedding
   - Directly addresses length disparity
   - Adds few parameters (hidden_size for query vector)

4. **Last token pooling**
   - Use final token representation (LLM-style)
   - Works well for decoder models

5. **Weighted combination of pooling strategies**
   - e.g., 0.5 * mean + 0.5 * max

### Projection Head (currently: single Linear 4096→768)

6. **MLP head**: Linear → ReLU → Linear
   - More expressive non-linear transformation
   - Cheap parameter increase

7. **Per-language-family heads**
   - Separate projections for CJK, Latin, Cyrillic, Arabic, Indic
   - Requires language detection/routing

8. **Mixture-of-experts head**
   - Soft routing based on input representation
   - More flexible than hard language families

9. **Deeper head with LayerNorm + residual**
   - Better gradient flow
   - More stable training

### LoRA on Base Model (currently frozen)

10. **LoRA on attention layers (q, v projections)**
    - Adapts representations themselves, not just projection
    - Typical rank: 8-64

11. **Language-family-specific LoRA adapters**
    - Different LoRA weights for different scripts
    - Most flexible but complex

12. **LoRA on last N layers only**
    - e.g., last 4 layers
    - Reduces parameter count, focuses on high-level features

13. **Low-rank (r=8) vs higher rank (r=64)**
    - Trade-off between expressivity and overfitting

### Combined Approaches

14. **LoRA + attention pooling**
    - Both representation and pooling are learnable

15. **LoRA + per-family heads**
    - Adapt base model + specialized projections

---

## Parameter Counts

- **Current (projection only):** ~3M params
- **With LoRA (r=16, last 4 layers):** ~10-20M additional
- **Attention pooling:** ~4K additional (just query vector)
- **MLP head:** ~6M (with hidden layer)

---

## Priority Recommendations

1. **Attention pooling** - directly addresses length disparity, minimal params
2. **LoRA (low rank)** - adapts representations, moderate param increase
3. **MLP head** - cheap expressivity gain

---

## Alternative Loss Functions

For reference, alternatives to current contrastive/InfoNCE loss:

- **Triplet loss**: anchor/positive/negative with margin
- **Multiple Negatives Ranking Loss (MNRL)**: sentence-transformers default
- **Hard negative mining**: mine similar but non-parallel pairs
- **Circle loss**: adaptive margins based on similarity
- **Knowledge distillation**: align to teacher model
