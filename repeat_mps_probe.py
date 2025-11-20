import torch
import time

def measure_mps_memory():
    """Get current MPS driver memory (macOS Metal)"""
    if torch.backends.mps.is_available():
        # Force synchronization to get accurate measurement
        torch.mps.synchronize()
        return torch.mps.driver_allocated_memory() / (1024**2)  # MB
    return 0

def test_sort_memory_leak(device="mps", num_iterations=100):
    """
    Mimic the repeat_mask pattern that causes per-step memory growth
    """
    print(f"Testing on device: {device}")
    print("=" * 80)
    
    # Parameters matching your use case
    B = 16384  # batch size
    M = 361    # history size
    
    # Pre-allocate tensors (simulating persistent state)
    hash_history = torch.randint(-1000, 1000, (B, M), dtype=torch.int32, device=device)
    new_hash = torch.randint(-1000, 1000, (B, M), dtype=torch.int32, device=device)
    move_count = torch.randint(100, M, (B,), dtype=torch.int32, device=device)
    
    INT32_MIN = torch.iinfo(torch.int32).min
    
    initial_mem = measure_mps_memory()
    print(f"Initial memory: {initial_mem:.1f} MB\n")
    
    mem_deltas = []
    
    for i in range(num_iterations):
        mem_before = measure_mps_memory()
        
        # ============ THE PROBLEMATIC PATTERN ============
        # 1. Create mask
        ar = torch.arange(M, device=device).view(1, -1)
        L = move_count.clamp_max(M).long()
        valid = (ar < L.view(-1, 1))
        
        # 2. Masked copy (creates new tensor each iteration)
        hist_masked = torch.where(valid, hash_history, 
                                  torch.full_like(hash_history, INT32_MIN))
        
        # 3. SORT - This is the likely culprit
        hist_sorted, _ = torch.sort(hist_masked, dim=1)
        
        # 4. Searchsorted
        idx = torch.searchsorted(hist_sorted, new_hash, right=True) - 1
        idx = idx.clamp_min(0)
        val = hist_sorted.gather(1, idx)
        
        # 5. Result
        is_repeat = (val == new_hash)
        # ================================================
        
        mem_after = measure_mps_memory()
        delta = mem_after - mem_before
        mem_deltas.append(delta)
        
        if i % 10 == 0 or delta > 1.0:  # Report every 10 steps or if jump > 1MB
            print(f"Iter {i:3d}: mem={mem_after:7.1f} MB  Δ={delta:+6.1f} MB")
    
    print("\n" + "=" * 80)
    print(f"Total memory growth: {mem_after - initial_mem:.1f} MB")
    print(f"Average per-step delta: {sum(mem_deltas)/len(mem_deltas):.2f} MB")
    print(f"Max single-step delta: {max(mem_deltas):.2f} MB")
    
    return mem_deltas


def test_alternative_no_sort(device="mps", num_iterations=100):
    """
    Test alternative approach without torch.sort()
    """
    print(f"\n{'='*80}")
    print("Testing ALTERNATIVE (no sort):")
    print("=" * 80)
    
    B = 16384
    M = 361
    
    hash_history = torch.randint(-1000, 1000, (B, M), dtype=torch.int32, device=device)
    new_hash = torch.randint(-1000, 1000, (B, M), dtype=torch.int32, device=device)
    move_count = torch.randint(100, M, (B,), dtype=torch.int32, device=device)
    
    # Pre-allocate buffer (reused across iterations)
    hist_buffer = torch.empty_like(hash_history)
    
    INT32_MIN = torch.iinfo(torch.int32).min
    
    initial_mem = measure_mps_memory()
    print(f"Initial memory: {initial_mem:.1f} MB\n")
    
    mem_deltas = []
    
    for i in range(num_iterations):
        mem_before = measure_mps_memory()
        
        # ============ ALTERNATIVE PATTERN ============
        # Copy into reusable buffer
        hist_buffer.copy_(hash_history)
        
        # In-place masking
        ar = torch.arange(M, device=device).view(1, -1)
        L = move_count.clamp_max(M).long()
        valid = (ar < L.view(-1, 1))
        hist_buffer.masked_fill_(~valid, INT32_MIN)
        
        # Still need sort, but on reused buffer
        hist_sorted, _ = torch.sort(hist_buffer, dim=1)
        
        idx = torch.searchsorted(hist_sorted, new_hash, right=True) - 1
        idx = idx.clamp_min(0)
        val = hist_sorted.gather(1, idx)
        is_repeat = (val == new_hash)
        # ============================================
        
        mem_after = measure_mps_memory()
        delta = mem_after - mem_before
        mem_deltas.append(delta)
        
        if i % 10 == 0 or delta > 1.0:
            print(f"Iter {i:3d}: mem={mem_after:7.1f} MB  Δ={delta:+6.1f} MB")
    
    print("\n" + "=" * 80)
    print(f"Total memory growth: {mem_after - initial_mem:.1f} MB")
    print(f"Average per-step delta: {sum(mem_deltas)/len(mem_deltas):.2f} MB")
    
    return mem_deltas


if __name__ == "__main__":
    # Ensure MPS is available
    if not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = "cpu"
    else:
        device = "mps"
    
    # Test original pattern
    deltas_original = test_sort_memory_leak(device=device, num_iterations=50)
    
    # Test alternative
    deltas_alternative = test_alternative_no_sort(device=device, num_iterations=50)
    
    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON:")
    print(f"Original approach: {sum(deltas_original):.1f} MB total growth")
    print(f"Alternative approach: {sum(deltas_alternative):.1f} MB total growth")
