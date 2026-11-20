# How KV Caching Speeds Up LLM Inference

## What KV Caching Is and Why It Matters

![Illustration of KV caching in transformer self‑attention](images/kv_cache_diagram.jpg)
*KV caching in transformer self‑attention: queries change per token while keys and values are cached and reused, reducing computation from quadratic to linear.*


Self‑attention in transformer models transforms each token into a query (Q), key (K), and value (V) matrix at every layer. When generating a new token, its query changes, but the keys and values of all previously generated tokens remain unchanged. KV caching stores those static K and V tensors after the first forward pass and re‑uses them in subsequent steps. By pulling the cached tensors directly into the attention computation, the model no longer needs to recompute the dot‑product between the new query and every earlier key. This eliminates the quadratic O(N²) cost associated with attending over all past positions, reducing the per‑token complexity to linear O(N), where N is the current context length. The result is a significant drop in latency, especially for long‑context generation, while keeping the same hardware and model parameters. Thus, KV caching is a simple yet powerful optimization that scales inference linearly with context length.

## Building a Minimal Inference Loop with KV Cache

Below is a compact, runnable sketch that demonstrates how to allocate, populate, and reuse a KV cache during token generation. It is written for PyTorch‑based transformers and assumes a model that accepts a `past_key_values` argument.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

def generate_with_cache(prompt, max_new_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    cache = {layer: (None, None) for layer in range(model.config.n_layer)}

    # First pass – compute keys/values
    outputs = model(input_ids, past_key_values=cache)
    logits = outputs.logits[:, -1, :]
    cache = outputs.past_key_values

    generated = input_ids
    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        # Forward with cached KV
        outputs = model(next_token, past_key_values=cache)
        logits = outputs.logits[:, -1, :]
        cache = outputs.past_key_values

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example usage
print(generate_with_cache("Once upon a time", max_new_tokens=30))
```

**Key points**

1. **Cache initialization** – `cache[layer] = (None, None)` creates placeholders for each transformer layer.  
2. **First pass** – `model.encode` is implicit in the forward call; the returned `past_key_values` are stored.  
3. **Subsequent tokens** – pass `past_key_values` to reuse the cached keys/values.  
4. **Cache update** – each step concatenates the new key/value to the existing tensors via `past_key_values`.  
5. **Device consistency** – the cache lives on the same device as the model (`cuda` in the example).  
6. **Batch & multi‑GPU** – the same pattern works; just ensure the tensors share the correct device and use `torch.cat` along `dim=1` for sequence length.  

This minimal loop shows the essential mechanics of KV caching in a production‑ready inference pipeline.
