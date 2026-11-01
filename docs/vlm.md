# vLLM memory components:
1. Model weights 
2. KV cache:
   2.1 Output-token generation (consume additional KV-cache memory)
3. Runtime overhead (CUDA)

# --gpu-memory-utilization 0.5:
1. vLLM loads the model, and then go to step 2
2. remaining GPU memory for KV cache.

# --max-model-len 32768:
1. Context Length=Input Tokens+Output Tokens

# --max-num-seqs 4:
1. controls how many requests vLLM can handle concurrently.
2. KV cache is shared as a memory pool, and allocated dynamically in blocks as requests arrive