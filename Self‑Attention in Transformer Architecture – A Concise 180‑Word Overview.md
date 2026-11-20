# Self‑Attention in Transformer Architecture

> **[IMAGE GENERATION FAILED]** Self‑attention computes context‑aware token representations via Q·Kᵀ scaling and weighted sum of V.
>
> **Alt:** Self‑attention mechanism flow diagram
>
> **Prompt:** Create a clear technical diagram of the self‑attention mechanism in a transformer model. Show token embeddings being linearly projected to Queries (Q), Keys (K), and Values (V). Illustrate the scaled dot‑product operation (Q multiplied by K transposed, scaled by sqrt(d_k)), the softmax to obtain attention weights, and the weighted sum of V producing context vectors. Use arrows to indicate data flow and label each step. Use a clean, professional style suitable for a technical blog.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\nPlease retry in 15h8m30.06574662s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://aistudio.google.com/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'description': 'Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'description': 'Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'description': 'Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '54510s'}]}}


## Core Mechanics of Self‑Attention
Grasp the mathematical flow that turns token embeddings into context‑aware representations. Each token vector is linearly projected to queries, keys, and values. The attention scores are computed as the scaled dot‑product of queries and keys, passed through a softmax to obtain weights, and then used to weight the values, yielding a new representation that mixes information from all positions.

## Minimal PyTorch Self‑Attention Sketch
```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

# Example usage
batch, seq_len, dim = 2, 5, 64
x = torch.randn(batch, seq_len, dim)
W_q = torch.nn.Linear(dim, dim)
W_k = torch.nn.Linear(dim, dim)
W_v = torch.nn.Linear(dim, dim)
Q = W_q(x)
K = W_k(x)
V = W_v(x)
output = scaled_dot_product_attention(Q, K, V)
```

## Performance & Scaling Considerations
The dominant cost is the O(seq_len²·dim) matrix multiplication for Q·Kᵀ, which grows quadratically with sequence length. Memory also scales with the full attention matrix. Mitigations include:
- **Sparse or local attention** to reduce quadratic terms.
- **Low‑rank approximations** (e.g., Linformer).
- **Chunking and streaming** for long sequences.
- **Efficient kernels** and mixed‑precision to accelerate the matmul.
These strategies help keep self‑attention feasible for longer inputs and larger models.