import torch
import time
from cs336_basics.model import scaled_dot_product_attention

batch_size = 8 # Fixed

d_model_list = [16, 32, 64, 128]
seq_len_list = [256, 1024, 4096, 8192, 16384]

n_iteration = 100
n_warmup = 10

device = torch.device("cuda")
sdpa = torch.compile(scaled_dot_product_attention)

for d_model in d_model_list:
    for seq_len in seq_len_list:
        try:
            Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

            seq = torch.arange(seq_len, device=device)
            causal_mask = seq.unsqueeze(0) >= seq.unsqueeze(1)

            for _ in range(n_warmup):
                out = scaled_dot_product_attention(Q, K, V, causal_mask)
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            tic = time.time()
            with torch.no_grad():
                for _ in range(n_iteration):
                    out = scaled_dot_product_attention(Q, K, V, causal_mask)
                    torch.cuda.synchronize()
            toc = time.time()

            forward_time = toc - tic
            forward_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            tic = time.time()
            for _ in range(n_iteration):
                Q_i = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                K_i = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                V_i = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                out = scaled_dot_product_attention(Q_i, K_i, V_i, causal_mask)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()

            toc = time.time()
            backward_time = toc - tic
            backward_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

            print(f"d_model={d_model}, seq_len={seq_len}: "
                        f"forward={forward_time:.4f}s, backward={backward_time:.4f}s, "
                        f"memory_before={forward_mem:.2f}MB, "
                        f"memory_after={backward_mem:.2f}MB")
        except torch.cuda.OutOfMemoryError:
                    print(f"d_model={d_model}, seq_len={seq_len}: OOM")
                    torch.cuda.empty_cache()