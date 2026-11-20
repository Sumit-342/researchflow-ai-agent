# Demystifying Self‑Attention in Transformer Architectures

## What Self‑Attention Actually Does

Self‑attention converts each token’s embedding into three vectors: a **query** \(Q\), a **key** \(K\), and a **value** \(V\). For a sequence of length \(n\) with embedding dimension \(d\), linear projections produce matrices \(Q, K, V \in \mathbb{R}^{n \times d_k}\) (often \(d_k = d_v = d\)).  

The core of self‑attention is the **similarity matrix** \(S = QK^\top\), whose entry \(s_{ij}\) is the dot‑product between the query of token \(i\) and the key of token \(j\). This measures how much token \(i\) should attend to token \(j\).  

Because dot‑products grow with dimensionality, we scale them by \(\frac{1}{\sqrt{d_k}}\) before the next step. The scaling factor keeps the softmax inputs in a moderate range, preventing vanishing or exploding gradients during training.  

Applying **softmax** row‑wise to the scaled matrix yields attention weights \(\alpha_{ij}\), a probability distribution over all tokens for each query.  

Finally, each token’s output is computed as a weighted sum of the value vectors:  
\[
\text{output}_i = \sum_{j=1}^{n} \alpha_{ij} V_j,
\]  
so the representation of token \(i\) aggregates information from the entire sequence, weighted by relevance, eliminating the need for recurrent connections.

## Implementing Scaled Dot‑Product Attention (MWE)

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q: torch.Tensor,
                                 K: torch.Tensor,
                                 V: torch.Tensor) -> torch.Tensor:
    """
    Compute a single attention head using scaled dot‑product attention.

    Args:
        Q: Queries of shape (batch, seq_len, d_k)
        K: Keys   of shape (batch, seq_len, d_k)
        V: Values of shape (batch, seq_len, d_v)

    Returns:
        Context vectors of shape (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)
    # 1️⃣ Raw scores: (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))

    # 2️⃣ Attention weights via softmax on the sequence dimension
    attn_weights = F.softmax(scores, dim=-1)

    # 3️⃣ Weighted sum of values
    context = torch.matmul(attn_weights, V)

    return context


# ----------------------------------------------------------------------
# Example usage with random tensors
batch, seq_len, d_k, d_v = 2, 5, 64, 64
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_v)

ctx = scaled_dot_product_attention(Q, K, V)

# Verify output shape matches (batch, seq_len, d_v)
assert ctx.shape == (batch, seq_len, d_v), f"Unexpected shape: {ctx.shape}"
print("Context shape:", ctx.shape)
```

The snippet follows the classic steps:

1. **Raw scores** are obtained with a matrix multiplication of queries and transposed keys, then scaled by \(1/\sqrt{d_k}\) to keep gradients stable.  
2. **Softmax** converts scores into a probability distribution across the sequence dimension, yielding attention weights.  
3. **Context vectors** result from multiplying these weights by the values tensor.

Running the script prints `Context shape: torch.Size([2, 5, 64])`, confirming that the function correctly processes a single attention head.

## Multi‑Head Attention: Parallelism and Performance

Transformer models gain expressive power by splitting the embedding dimension into `h` parallel attention heads. This is achieved by reshaping the query, key, and value tensors from a shape of *(batch, seq_len, d_model)* to *(batch, heads, seq_len, d_k)*, where `d_k = d_model / h`. Each head operates on a lower‑dimensional sub‑space, allowing the model to capture distinct relational patterns simultaneously.

Once reshaped, scaled dot‑product attention is executed independently on each head. In practice the operation is fully vectorized across the `heads` dimension, so a single matrix‑multiply kernel computes all heads in parallel. This yields the same computational complexity as a single head but distributes the work, improving hardware utilization on GPUs and TPUs.

The outputs of the `h` heads—each of shape *(batch, seq_len, d_k)*—are concatenated along the last dimension to reconstruct a tensor of shape *(batch, seq_len, d_model)*. A final linear projection then mixes information across heads, restoring the original model dimension.

From a cost perspective, each head contributes `O(seq_len²·d_k)` floating‑point operations. Because `d_k = d_model / h`, the total FLOPs scale linearly with the number of heads: `O(seq_len²·d_model·h / h) = O(seq_len²·d_model)`. The arithmetic workload remains constant, but the constant factor grows with `h` due to additional matrix‑multiply launches and memory traffic.

Memory usage rises more sharply. Storing separate Q, K, V tensors per head multiplies the per‑token memory footprint by `h`. For long sequences this can dominate GPU memory, limiting batch size or requiring gradient checkpointing.

To mitigate bandwidth pressure, modern implementations adopt FlashAttention or kernel‑fusion techniques. FlashAttention computes the attention matrix in a cache‑friendly, tiled fashion, reducing the number of reads/writes to DRAM. Kernel fusion merges the QKV projection, attention scoring, and output projection into a single GPU kernel, cutting launch overhead and improving throughput. These optimizations preserve the parallelism benefits of multi‑head attention while keeping FLOPs and memory demands in check.

## Debugging & Edge Cases in Self‑Attention

- **Validate the attention mask** – The mask must have shape `[batch, 1, 1, seq_len]` (or the equivalent broadcastable form) and be a boolean or `torch.float` tensor where padding positions are `0`. A mismatched shape or dtype will let padding influence the softmax, producing spurious attention to padded tokens. Print `mask.shape` and `mask.dtype` before the `torch.nn.functional.softmax` call to catch this early.  

- **Detect NaNs after softmax** – Extreme dot‑product scores can overflow, yielding `NaN` values. After `softmax`, run `torch.isnan(attn).any()`; if true, clamp the raw scores (`torch.clamp(scores, min=-MAX, max=MAX)`) or explicitly cast to `float32` before softmax: `torch.nn.functional.softmax(scores.to(torch.float32), dim=-1)`.  

- **Profile attention latency** – Wrap the Q·K multiplication in `torch.autograd.profiler.profile()` to identify bottlenecks. Look for unusually high `cpu_time_total` or `cuda_time_total` in the `matmul` node; consider fused kernels or lower‑precision matmul if it dominates runtime.  

- **Edge case: sequence length = 1** – With a single token, softmax must return `1.0`. Add a unit test: `assert torch.allclose(attn.squeeze(), torch.ones_like(attn.squeeze()))`. This guards against division‑by‑zero or masking errors.  

- **Mixed‑precision scaling** – When `torch.cuda.amp` is active, compute the scaling factor (e.g., `sqrt(d_k)`) in FP32: `scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))`. Using FP16 can cause underflow, leading to all‑zero attention scores.  

- **Log attention weight distribution** – Periodically record a histogram of the attention matrix (e.g., `torch.histc(attn, bins=50)`) to spot collapsed distributions (all mass on a single head) or overly uniform patterns, both of which indicate training pathologies.
