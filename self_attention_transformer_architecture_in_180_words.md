# Self‑Attention Transformer Architecture in 180 Words

## Core Concepts

![High-level diagram of Transformer encoder architecture](images/transformer_architecture.jpg)
*Figure 1: Transformer encoder architecture showing embedding, multi‑head self‑attention, feed‑forward, residual connections, layer normalization, and output head.*


1. **Embedding**: converts tokens into dense vectors and adds positional encodings.  
2. **Encoder Stack**: a series of layers; each applies multi‑head self‑attention followed by a position‑wise feed‑forward network, wrapped with residual connections and layer normalization.  
3. **Output Head**: projects the final encoder output to the target vocabulary logits.  

Positional encodings supply token order to the otherwise permutation‑invariant self‑attention mechanism.

## Self‑Attention Mechanics

Queries, keys, and values are linear projections: Q=WX_q, K=WX_k, V=WX_v. The scaled dot‑product attention computes A=softmax(QKᵀ/√d_k)V. In practice we split the embedding into h heads, each computing its own A, then concatenate. A minimal single‑head PyTorch module: The module inherits nn.Module, defines weight matrices W_q, W_k, W_v, and a forward that returns torch.matmul(F.softmax(torch.matmul(Q, K.transpose(-2,-1))/sqrt(d_k), V)). This simple block demonstrates core attention mechanics.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** 0.5

    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        return torch.matmul(attn, V)
```

## Practical Implications

Self‑attention’s quadratic O(n²) memory cost scales poorly with sequence length n. For long inputs, sparse or linear‑attention variants cut this overhead. Adding more heads boosts expressivity but also raises compute. Deep stacks often suffer vanishing gradients, so layer‑norm or residual scaling is essential to maintain training stability. In practice, choose single‑head linear‑attention for speed, or multi‑head deep stacks with normalization.
