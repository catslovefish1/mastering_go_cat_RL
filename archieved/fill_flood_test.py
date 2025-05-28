import torch
import torch.nn.functional as F
from torch import Tensor

def _flood_fill_morphology(self, seeds: Tensor, mask: Tensor) -> Tensor:
    """
    Drop-in replacement for _flood_fill using convolution/morphology approach.
    
    This method can directly replace the original _flood_fill method in TensorBoard class.
    It produces identical results but uses convolution instead of slicing.
    
    Args:
        seeds: BoardTensor of shape (B, H, W) marking starting positions
        mask: BoardTensor of shape (B, H, W) defining valid expansion areas
        
    Returns:
        BoardTensor containing all positions in the connected groups
    """
    # Early exit - same as original
    if not seeds.any():
        return seeds
    
    # Create structuring element for 4-connectivity
    # This replaces the four slicing operations in the original
    selem = torch.tensor(
        [[0, 1, 0],
         [1, 0, 1],  # Center is 0 for neighbor finding
         [0, 1, 0]], 
        dtype=torch.float32,
        device=seeds.device  # Important: use same device as input
    ).view(1, 1, 3, 3)
    
    # Initialize groups - same as original
    groups = seeds.clone()
    
    # Convert to float for convolution
    groups_float = groups.float()
    mask_float = mask.float()
    
    # Main loop - same structure as original
    for _ in range(self.max_flood_iterations):
        # Find neighbors using convolution instead of slicing
        # This replaces the four slicing operations:
        # self._flood_expanded[:, 1:, :] |= groups[:, :-1, :]  etc.
        neighbors = F.conv2d(
            groups_float.unsqueeze(1),  # Add channel dimension
            selem,
            padding=1  # Same padding to maintain size
        ).squeeze(1)  # Remove channel dimension
        
        # Threshold to binary (any neighbor contact = expansion)
        # Equivalent to the OR operations in original
        expanded = neighbors > 0
        
        # Apply mask constraint - same as original
        # self._flood_expanded &= mask
        expanded = expanded & mask
        
        # Remove already visited - same as original  
        # self._flood_expanded &= ~groups
        expanded = expanded & ~groups
        
        # Check convergence - same as original
        if not expanded.any():
            break
        
        # Update groups - same as original
        # groups |= self._flood_expanded
        groups = groups | expanded
        groups_float = groups.float()  # Update float version
    
    return groups


def _flood_fill_original(self, seeds: Tensor, mask: Tensor) -> Tensor:
    """Original flood fill implementation for comparison"""
    if not seeds.any():
        return seeds
    
    groups = seeds.clone()
    
    # Pre-allocate work buffer (simulating the original's approach)
    flood_expanded = torch.zeros_like(seeds, dtype=torch.bool)
    
    for _ in range(self.max_flood_iterations):
        # Clear expanded buffer
        flood_expanded.zero_()
        
        # Get neighbors using slicing
        # Top neighbors
        flood_expanded[:, 1:, :] |= groups[:, :-1, :]
        # Bottom neighbors  
        flood_expanded[:, :-1, :] |= groups[:, 1:, :]
        # Left neighbors
        flood_expanded[:, :, 1:] |= groups[:, :, :-1]
        # Right neighbors
        flood_expanded[:, :, :-1] |= groups[:, :, 1:]
        
        # Mask to valid positions and exclude already visited
        flood_expanded &= mask
        flood_expanded &= ~groups
        
        # Check if we found new positions
        if not flood_expanded.any():
            break
        
        # Add to groups
        groups |= flood_expanded
    
    return groups


def verify_flood_fill_equivalence():
    """Comprehensive test to verify both implementations produce identical results"""
    
    # Mock class to hold the max_flood_iterations attribute
    class MockBoard:
        def __init__(self):
            self.max_flood_iterations = 100
    
    mock_board = MockBoard()
    
    print("Testing Flood Fill Implementations Equivalence")
    print("=" * 60)
    
    test_cases = []
    
    # Test Case 1: Simple connected component
    test_cases.append({
        'name': 'Simple Connected Component',
        'size': (1, 7, 7),
        'seeds': [(0, 0, 0)],
        'mask_points': [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 1)]
    })
    
    # Test Case 2: Complex snake pattern
    test_cases.append({
        'name': 'Snake Pattern',
        'size': (1, 9, 9),
        'seeds': [(0, 0, 0)],
        'mask_points': [
            (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
            (0, 1, 3), (0, 2, 3), (0, 3, 3), (0, 3, 2),
            (0, 3, 1), (0, 4, 1), (0, 5, 1), (0, 5, 2),
            (0, 5, 3), (0, 5, 4), (0, 5, 5)
        ]
    })
    
    # Test Case 3: Multiple batches
    test_cases.append({
        'name': 'Multiple Batches',
        'size': (4, 5, 5),
        'seeds': [(0, 2, 2), (1, 0, 0), (2, 1, 1), (3, 3, 3)],
        'mask_points': [
            # Batch 0 - cross pattern
            (0, 2, 2), (0, 1, 2), (0, 3, 2), (0, 2, 1), (0, 2, 3),
            # Batch 1 - L shape
            (1, 0, 0), (1, 1, 0), (1, 2, 0), (1, 2, 1),
            # Batch 2 - square
            (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2),
            # Batch 3 - single point
            (3, 3, 3)
        ]
    })
    
    # Test Case 4: Empty seeds
    test_cases.append({
        'name': 'Empty Seeds',
        'size': (2, 5, 5),
        'seeds': [],
        'mask_points': [(0, 1, 1), (0, 2, 2), (1, 3, 3)]
    })
    
    # Test Case 5: Full board
    test_cases.append({
        'name': 'Full Board',
        'size': (1, 4, 4),
        'seeds': [(0, 0, 0)],
        'mask_points': [(0, i, j) for i in range(4) for j in range(4)]
    })
    
    # Test Case 6: Disconnected regions (should not connect)
    test_cases.append({
        'name': 'Disconnected Regions',
        'size': (1, 7, 7),
        'seeds': [(0, 1, 1)],
        'mask_points': [
            # First region
            (0, 1, 1), (0, 1, 2), (0, 2, 1), (0, 2, 2),
            # Second region (disconnected)
            (0, 4, 4), (0, 4, 5), (0, 5, 4), (0, 5, 5)
        ]
    })
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print("-" * 40)
        
        # Create tensors
        size = test_case['size']
        seeds = torch.zeros(size, dtype=torch.bool)
        mask = torch.zeros(size, dtype=torch.bool)
        
        # Set seeds
        for coord in test_case['seeds']:
            seeds[coord] = True
        
        # Set mask
        for coord in test_case['mask_points']:
            mask[coord] = True
        
        # Run both implementations
        result_original = _flood_fill_original(mock_board, seeds, mask)
        result_morphology = _flood_fill_morphology(mock_board, seeds, mask)
        
        # Compare results
        if torch.equal(result_original, result_morphology):
            print("✓ PASS: Results are identical")
            print(f"  Seeds: {len(test_case['seeds'])} points")
            print(f"  Mask points: {len(test_case['mask_points'])} points")
            print(f"  Filled points: {result_original.sum().item()} points")
        else:
            print("✗ FAIL: Results differ!")
            all_passed = False
            
            # Show differences
            diff = (result_original.float() - result_morphology.float()).abs()
            diff_points = diff.nonzero(as_tuple=False)
            print(f"  Number of differences: {len(diff_points)}")
            if len(diff_points) <= 10:
                print(f"  Difference locations: {diff_points.tolist()}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    # Create a larger test case for timing
    import time
    
    # 19x19 board with complex pattern
    batch_size = 8
    board_size = 19
    
    # Create random connected components
    torch.manual_seed(42)
    seeds = torch.zeros(batch_size, board_size, board_size, dtype=torch.bool)
    mask = torch.rand(batch_size, board_size, board_size) > 0.6  # ~40% filled
    
    # Set random seeds
    for b in range(batch_size):
        seeds[b, torch.randint(0, board_size, (1,)), torch.randint(0, board_size, (1,))] = True
    
    # Time original
    start = time.time()
    for _ in range(100):
        result_original = _flood_fill_original(mock_board, seeds.clone(), mask)
    time_original = time.time() - start
    
    # Time morphology
    start = time.time()
    for _ in range(100):
        result_morphology = _flood_fill_morphology(mock_board, seeds.clone(), mask)
    time_morphology = time.time() - start
    
    print(f"Original (slicing): {time_original:.3f} seconds for 100 iterations")
    print(f"Morphology (conv):  {time_morphology:.3f} seconds for 100 iterations")
    print(f"Ratio: {time_morphology/time_original:.2f}x")
    
    if all_passed:
        print("\n✓ All tests passed! The morphology implementation is functionally equivalent.")
    else:
        print("\n✗ Some tests failed! The implementations are not equivalent.")
    
    return all_passed


def demo_visual_comparison():
    """Visual demonstration of both methods on a simple example"""
    
    class MockBoard:
        def __init__(self):
            self.max_flood_iterations = 100
    
    mock_board = MockBoard()
    
    print("\nVisual Comparison Demo")
    print("=" * 60)
    
    # Create a simple L-shaped pattern
    seeds = torch.zeros(1, 5, 5, dtype=torch.bool)
    seeds[0, 0, 0] = True  # Start point
    
    mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    # L shape
    mask[0, 0, 0] = True
    mask[0, 1, 0] = True
    mask[0, 2, 0] = True
    mask[0, 3, 0] = True
    mask[0, 3, 1] = True
    mask[0, 3, 2] = True
    
    print("Mask (L-shaped region):")
    for i in range(5):
        row = ""
        for j in range(5):
            row += "■ " if mask[0, i, j] else ". "
        print(row)
    
    print("\nSeed position: (0, 0)")
    
    result_original = _flood_fill_original(mock_board, seeds, mask)
    result_morphology = _flood_fill_morphology(mock_board, seeds, mask)
    
    print("\nOriginal method result:")
    for i in range(5):
        row = ""
        for j in range(5):
            row += "● " if result_original[0, i, j] else ". "
        print(row)
    
    print("\nMorphology method result:")
    for i in range(5):
        row = ""
        for j in range(5):
            row += "● " if result_morphology[0, i, j] else ". "
        print(row)
    
    print(f"\nResults identical: {torch.equal(result_original, result_morphology)}")


if __name__ == "__main__":
    # Run verification
    verify_flood_fill_equivalence()
    
    # Run visual demo
    demo_visual_comparison()
    
    print("\n" + "=" * 60)
    print("USAGE: To use in TensorBoard class, simply add this method:")
    print("=" * 60)
    print("""
# Add this method to your TensorBoard class:
def _flood_fill(self, seeds: BoardTensor, mask: BoardTensor) -> BoardTensor:
    return _flood_fill_morphology(self, seeds, mask)

# Or directly replace the existing _flood_fill method with the 
# _flood_fill_morphology implementation above.
""")