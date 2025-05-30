import torch
import time

def benchmark_counting(batch_size=512, num_captures=1000, device='mps'):
    # Setup test data
    batch_idx = torch.randint(0, batch_size, (num_captures,), device=device)
    
    # Warmup
    for _ in range(10):
        torch.bincount(batch_idx, minlength=batch_size)
        capture_counts = torch.zeros(batch_size, device=device, dtype=torch.long)  # Fix: match dtype
        capture_counts.scatter_add_(0, batch_idx, torch.ones_like(batch_idx))
    
    # Benchmark bincount
    if device == 'mps':
        torch.mps.synchronize()  # Fix: use mps sync
    elif device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        counts1 = torch.bincount(batch_idx, minlength=batch_size)
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    bincount_time = time.time() - start
    
    # Benchmark scatter_add
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        counts2 = torch.zeros(batch_size, device=device, dtype=torch.long)  # Fix: match dtype
        counts2.scatter_add_(0, batch_idx, torch.ones_like(batch_idx))
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    scatter_time = time.time() - start
    
    print(f"Bincount: {bincount_time:.4f}s")
    print(f"Scatter_add: {scatter_time:.4f}s")
    print(f"Bincount is {scatter_time/bincount_time:.2f}x faster")

# Run benchmark
benchmark_counting()