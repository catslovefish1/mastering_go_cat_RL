# Understanding Tensor Reshaping in PyTorch

import torch

# ============================================
# PART 1: What is reshaping?
# ============================================
# Reshaping changes how data is organized without changing the data itself
# Think of it like rearranging books on shelves

# Example: You have 12 books
books = torch.arange(12)  # Creates [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print("Original 12 books in a line:")
print(books)
print(f"Shape: {books.shape}")  # Shape: torch.Size([12])

# You can arrange them as:
# - 2 shelves with 6 books each
shelves_2x6 = books.view(2, 6)
print("\n2 shelves × 6 books:")
print(shelves_2x6)
print(f"Shape: {shelves_2x6.shape}")  # Shape: torch.Size([2, 6])

# - 3 shelves with 4 books each
shelves_3x4 = books.view(3, 4)
print("\n3 shelves × 4 books:")
print(shelves_3x4)
print(f"Shape: {shelves_3x4.shape}")  # Shape: torch.Size([3, 4])

# - 4 shelves with 3 books each
shelves_4x3 = books.view(4, 3)
print("\n4 shelves × 3 books:")
print(shelves_4x3)
print(f"Shape: {shelves_4x3.shape}")  # Shape: torch.Size([4, 3])

# ============================================
# PART 2: The Golden Rule
# ============================================
# The total number of elements MUST stay the same!
# 12 = 2×6 = 3×4 = 4×3 = 6×2 = 12×1 = 1×12

# This will work:
ok_reshape = books.view(2, 6)  # 2 × 6 = 12 ✓

# This will FAIL:
try:
    bad_reshape = books.view(2, 5)  # 2 × 5 = 10 ≠ 12 ✗
except RuntimeError as e:
    print(f"\nError: {e}")
    print("Can't reshape 12 elements into 2×5 (only 10 spots!)")

# ============================================
# PART 3: The Error in Your Code
# ============================================
# Your code had:
history_factor = 10
board_size = 5

# It created a tensor with this many elements:
stones_size = history_factor * board_size * board_size  # 10 × 5 × 5 = 250
print(f"\nYour tensor had {stones_size} elements")

# Then tried to reshape to:
target_shape = (2, board_size * board_size)  # (2, 25)
needed_elements = 2 * 25  # = 50
print(f"But shape {target_shape} only needs {needed_elements} elements")
print(f"250 ≠ 50, so it fails!")

# ============================================
# PART 4: Visual Example with Go Board
# ============================================
# Let's use a tiny 3×3 Go board as example
board_size = 3

# Correct approach: 2 colors × 9 positions = 18 elements
correct_size = 2 * board_size * board_size  # 2 × 9 = 18
zobrist_values = torch.randint(0, 1000, (correct_size,))
print(f"\nCreated {correct_size} random hash values")

# Now we can reshape to (2, 9) - one row per color
zobrist_by_color = zobrist_values.view(2, board_size * board_size)
print(f"Reshaped to {zobrist_by_color.shape}:")
print(f"Black hash values: {zobrist_by_color[0]}")  # 9 values for black stones
print(f"White hash values: {zobrist_by_color[1]}")  # 9 values for white stones

# ============================================
# PART 5: Common Reshape Patterns
# ============================================

# Pattern 1: Flatten (make everything 1D)
matrix = torch.arange(12).view(3, 4)
flattened = matrix.view(-1)  # -1 means "figure out this dimension"
print(f"\nMatrix {matrix.shape} → Flattened {flattened.shape}")

# Pattern 2: Add a dimension
vector = torch.arange(5)
print(f"\nOriginal vector: {vector.shape}")
row_vector = vector.view(1, 5)  # Make it a row
print(f"As row: {row_vector.shape}")
col_vector = vector.view(5, 1)  # Make it a column  
print(f"As column: {col_vector.shape}")

# Pattern 3: Batch processing
# Often we have (batch_size, height, width) for images
batch_size = 4
height = 28
width = 28
images = torch.randn(batch_size, height, width)
print(f"\nBatch of images: {images.shape}")

# Flatten each image for a neural network
flattened_batch = images.view(batch_size, -1)  # -1 = height×width
print(f"Flattened for network: {flattened_batch.shape}")

# ============================================
# DEBUGGING TIP: Always check dimensions!
# ============================================
def safe_reshape(tensor, new_shape):
    """Helper to debug reshape operations"""
    old_elements = tensor.numel()  # Total number of elements
    new_elements = 1
    for dim in new_shape:
        if dim != -1:  # -1 is special "infer this" value
            new_elements *= dim
    
    print(f"\nReshaping from {tensor.shape} to {new_shape}")
    print(f"Old total: {old_elements} elements")
    print(f"New total: {new_elements} elements")
    
    if old_elements == new_elements:
        print("✓ Reshape will work!")
        return tensor.view(new_shape)
    else:
        print(f"✗ Reshape will FAIL! {old_elements} ≠ {new_elements}")
        return None

# Example usage:
data = torch.arange(24)
safe_reshape(data, (4, 6))   # Will work: 24 = 4×6
safe_reshape(data, (5, 5))   # Will fail: 24 ≠ 25