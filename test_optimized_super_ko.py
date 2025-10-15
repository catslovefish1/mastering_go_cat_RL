#!/usr/bin/env python3
"""
Test script to compare original vs optimized super-ko filtering.
Shows how to integrate the optimized methods into existing TensorBoard class.
"""

import torch
import time
from typing import Dict
from engine.tensor_native import TensorBoard
from engine.tensor_native_optimized import OptimizedSuperKo

def benchmark_super_ko_methods():
    """Compare performance of original vs optimized super-ko filtering."""
    
    # Setup parameters
    batch_size = 4
    board_size = 19
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing on device: {device}")
    print(f"Batch size: {batch_size}, Board size: {board_size}x{board_size}")
    print("-" * 60)
    
    # Create board instance
    board = TensorBoard(
        batch_size=batch_size,
        board_size=board_size,
        device=device,
        enable_super_ko=True,
        enable_timing=False
    )
    
    # Simulate some game state
    N2 = board_size * board_size
    
    # Create mock legal mask and capture info
    legal_mask = torch.rand(batch_size, board_size, board_size, device=device) > 0.7
    
    # Mock capture stone mask (sparse - most moves don't capture)
    capture_stone_mask = torch.rand(batch_size, N2, N2, device=device) > 0.95
    
    info = {"capture_stone_mask": capture_stone_mask}
    
    # Add some history
    board.move_count[:] = 50
    for i in range(50):
        board.hash_history[:, i] = torch.randint(0, 2**31, (batch_size,), dtype=torch.int32, device=device)
    
    print("1. Testing Original Implementation")
    print("-" * 40)
    
    # Warm up
    for _ in range(3):
        _ = board._filter_super_ko_vectorized(legal_mask.clone(), info)
    
    # Time original
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    for _ in range(10):
        result_original = board._filter_super_ko_vectorized(legal_mask.clone(), info)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_original = (time.time() - start) / 10
    
    print(f"Time per call: {time_original*1000:.2f} ms")
    print(f"Memory estimate: ~{batch_size * N2 * N2 * 4 / (1024**2):.1f} MB for dense int32 tensors")
    
    print("\n2. Testing Optimized Implementation (Sparse)")
    print("-" * 40)
    
    # Test sparse method
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    for _ in range(10):
        result_sparse = OptimizedSuperKo.filter_super_ko_optimized(
            legal_mask=legal_mask.clone(),
            capture_stone_mask=capture_stone_mask,
            current_hash=board.current_hash,
            hash_history=board.hash_history,
            move_count=board.move_count,
            zobrist_table=board.Zpos,
            zobrist_transposed=board.ZposT,
            current_player=board.current_player,
            use_sparse=True,
            history_chunk_size=64
        )
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_sparse = (time.time() - start) / 10
    
    print(f"Time per call: {time_sparse*1000:.2f} ms")
    print(f"Speedup: {time_original/time_sparse:.2f}x")
    
    print("\n3. Testing Optimized Implementation (Chunked)")
    print("-" * 40)
    
    # Test chunked method
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    for _ in range(10):
        result_chunked = OptimizedSuperKo.filter_super_ko_optimized(
            legal_mask=legal_mask.clone(),
            capture_stone_mask=capture_stone_mask,
            current_hash=board.current_hash,
            hash_history=board.hash_history,
            move_count=board.move_count,
            zobrist_table=board.Zpos,
            zobrist_transposed=board.ZposT,
            current_player=board.current_player,
            use_sparse=False,  # Use chunked method
            history_chunk_size=64
        )
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_chunked = (time.time() - start) / 10
    
    print(f"Time per call: {time_chunked*1000:.2f} ms")
    print(f"Speedup: {time_original/time_chunked:.2f}x")
    
    # Verify results match
    print("\n4. Verification")
    print("-" * 40)
    
    match_sparse = torch.allclose(result_original, result_sparse)
    match_chunked = torch.allclose(result_original, result_chunked)
    
    print(f"Sparse method matches original: {match_sparse}")
    print(f"Chunked method matches original: {match_chunked}")
    
    return result_original, result_sparse, result_chunked


def show_integration_example():
    """Show how to integrate the optimized method into existing TensorBoard class."""
    
    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE")
    print("=" * 60)
    
    example_code = '''
# In your engine/tensor_native.py, replace the _filter_super_ko_vectorized method:

from engine.tensor_native_optimized import OptimizedSuperKo

class TensorBoard(torch.nn.Module):
    # ... existing code ...
    
    @timed_method
    def _filter_super_ko_vectorized(self, legal_mask: BoardTensor, info: Dict) -> BoardTensor:
        """
        Optimized super-ko filter using sparse or chunked computation.
        """
        # Use sparse method for boards with few captures (typical case)
        # Use chunked method for boards with many captures
        
        capture_density = info["capture_stone_mask"].float().mean()
        use_sparse = capture_density < 0.1  # Threshold can be tuned
        
        return OptimizedSuperKo.filter_super_ko_optimized(
            legal_mask=legal_mask,
            capture_stone_mask=info["capture_stone_mask"],
            current_hash=self.current_hash,
            hash_history=self.hash_history,
            move_count=self.move_count,
            zobrist_table=self.Zpos,
            zobrist_transposed=self.ZposT,
            current_player=self.current_player,
            use_sparse=use_sparse,
            history_chunk_size=64  # Can be tuned based on memory constraints
        )
'''
    
    print(example_code)
    
    print("\nKey Improvements:")
    print("-" * 40)
    print("1. **Memory Efficiency**: Avoids creating dense B*N2*N2 int32 tensors")
    print("2. **Type Consistency**: Uses int32 throughout (no unnecessary int64)")
    print("3. **Sparse Operations**: Leverages sparsity of capture masks")
    print("4. **Chunked Processing**: Reduces peak memory usage")
    print("5. **Optimized XOR Reduction**: Uses reshape instead of slicing")
    print("\nRecommendations:")
    print("-" * 40)
    print("• Use sparse method when capture_stone_mask is < 10% dense (typical)")
    print("• Adjust history_chunk_size based on available memory (32-128 typical)")
    print("• Consider caching ZposT at initialization (already done)")
    print("• For very large boards (>19x19), always use sparse method")


if __name__ == "__main__":
    try:
        # Run benchmark
        results = benchmark_super_ko_methods()
        
        # Show integration guide
        show_integration_example()
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("\nMake sure you have the required files:")
        print("  - engine/tensor_native.py")
        print("  - engine/tensor_native_optimized.py")
        print("  - utils/shared.py")
