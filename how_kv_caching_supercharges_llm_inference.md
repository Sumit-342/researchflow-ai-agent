# How KV Caching Supercharges LLM Inference

## What is KV caching and why it matters

KV caching stores the attention keys and values produced by each transformer layer for every token that has already been processed. Instead of recomputing these matrices from scratch when a new token is generated, the model reuses the cached keys (K) and values (V) and only computes the fresh query (Q) for the incoming token. In the standard self‑attention equation \( \text{Attention}(Q,K,V)=\text{softmax}(QK^{\top}/\sqrt{d})V \), the \(K\) and \(V\) terms remain constant across a generation step, so the expensive matrix multiplications involving past tokens can be skipped. The cache typically lives for the duration of a single generation session; its memory footprint grows linearly with the sequence length (O(L × d) per layer). Importantly, KV caching does not modify the model’s learned weights—it merely changes the inference data path, yielding faster latency and lower compute without affecting model accuracy.

## Step‑by‑step flow of cached inference

1. **Tokenization & embedding** – The input text is split into tokens, each mapped to an embedding vector. The first transformer layer processes these embeddings, generating the initial query, key, and value tensors that populate the KV cache for layer 1 (and similarly for deeper layers).

2. **Per‑layer cache lookup** – For every subsequent token, each layer retrieves the stored keys and values, concatenates the newly computed ones to the end of the existing cache, and then uses the combined set for attention. This avoids recomputing past keys/values.

3. **Attention score computation** – Queries (Q) for the current token are multiplied by the transpose of the full key matrix (Kᵀ), which now includes both cached and fresh keys. The resulting scores are scaled and soft‑maxed to weight the values.

4. **Cache update** – After the layer produces the new hidden state, its corresponding key and value are appended to the cache, ready for the next token’s attention pass.

5. **Batch handling** – With batch size 1, cache indexing is straightforward—each token’s KV pair occupies the next slot. In larger batches, offsets must be tracked so each sequence’s cache is correctly aligned, preventing cross‑talk between examples.

## Minimal code sketch for a cached transformer loop

```python
import torch
from model import TransformerLM   # your transformer model

model = TransformerLM().to("cuda")
model.eval()

# ---- KV cache initialization (one empty list per layer) ----
kv_cache = {
    "keys":   [[] for _ in range(model.num_layers)],
    "values": [[] for _ in range(model.num_layers)],
}

input_ids = torch.tensor([[tokenizer.bos_token_id]], device="cuda")
generated = []

for step in range(max_len):
    # Forward pass returns logits and the newly computed KV pairs
    logits, new_keys, new_values = model.forward(
        input_ids,
        past_keys=kv_cache["keys"],
        past_values=kv_cache["values"]
    )

    # ---- Update cache: store new KV for each layer ----
    for i in range(model.num_layers):
        kv_cache["keys"][i].append(new_keys[i].detach())
        kv_cache["values"][i].append(new_values[i].detach())

    # Move cache tensors to the same device as the model (if needed)
    # for i in range(model.num_layers):
    #     kv_cache["keys"][i]   = kv_cache["keys"][i].to(model.device)
    #     kv_cache["values"][i] = kv_cache["values"][i].to(model.device)

    # Sample next token
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(next_token.item())
    input_ids = next_token

    # ---- Optional: truncate/roll cache to keep a fixed window size ----
    if len(kv_cache["keys"][0]) > cache_window:
        for i in range(model.num_layers):
            kv_cache["keys"][i]   = kv_cache["keys"][i][-cache_window:]
            kv_cache["values"][i] = kv_cache["values"][i][-cache_window:]

    if next_token.item() == tokenizer.eos_token_id:
        break

print("Generated sequence:", generated)
```

*The cache stores per‑layer key/value tensors across decoding steps, enabling the model to attend only to previously computed states. Truncating the cache prevents unbounded memory growth for long generations.*

## Performance, memory, and edge‑case considerations

KV caching typically cuts inference latency by **2–3×** for 1 k‑token continuations on modern GPUs, because each new token reuses previously computed key/value matrices instead of recomputing the full attention stack. The memory cost grows linearly with the cached context: each token adds **O(L · layers · heads · head_dim)** bytes (L = sequence length). When the accumulated cache approaches GPU memory limits, performance degrades; practitioners must either offload older KV pairs to host RAM, truncate the context, or employ sliding‑window strategies to keep the active window within bounds.  

Edge cases can break naïve caching: padding tokens still occupy KV slots, variable‑length batches produce mis‑aligned caches, and custom attention masks (e.g., causal or block‑sparse masks) may invalidate simple concatenation of new keys/values.  

To ensure the cache behaves as expected, profile with tools such as **torch.profiler** or **nsight systems**, measuring cache hit ratios, memory pressure, and any unexpected attention‑mask overhead. These diagnostics help balance speed gains against memory consumption and identify failure modes early.
