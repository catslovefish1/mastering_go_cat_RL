import torch
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import numpy as np

class FloodFillAnalyzer:
    """Analyze how many iterations flood fill takes in different scenarios"""
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.max_flood_iterations = min(board_size * board_size // 4, 100)
        
        # Pre-allocate work buffer
        self._flood_expanded = torch.zeros((1, board_size, board_size), dtype=torch.bool)
    
    def _flood_fill_with_count(self, seeds: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Modified flood fill that returns iteration count"""
        if not seeds.any():
            return seeds, 0
            
        groups = seeds.clone()
        iteration_count = 0
        
        for iteration in range(self.max_flood_iterations):
            iteration_count += 1
            
            # Clear expanded buffer
            self._flood_expanded.zero_()
            
            # Get neighbors using slicing
            # Top neighbors
            self._flood_expanded[:, 1:, :] |= groups[:, :-1, :]
            # Bottom neighbors  
            self._flood_expanded[:, :-1, :] |= groups[:, 1:, :]
            # Left neighbors
            self._flood_expanded[:, :, 1:] |= groups[:, :, :-1]
            # Right neighbors
            self._flood_expanded[:, :, :-1] |= groups[:, :, 1:]
            
            # Mask to valid positions and exclude already visited
            self._flood_expanded &= mask
            self._flood_expanded &= ~groups
            
            # Check if we found new positions
            if not self._flood_expanded.any():
                break
            
            # Add to groups in-place
            groups |= self._flood_expanded
        
        return groups, iteration_count
    
    def create_test_patterns(self) -> Dict[str, torch.Tensor]:
        """Create various test patterns to analyze iteration counts"""
        patterns = {}
        size = self.board_size
        
        # 1. Single stone (trivial case)
        single = torch.zeros(1, size, size, dtype=torch.bool)
        single[0, size//2, size//2] = True
        patterns['single_stone'] = single
        
        # 2. Straight line (horizontal)
        line_h = torch.zeros(1, size, size, dtype=torch.bool)
        line_h[0, size//2, :] = True
        patterns['horizontal_line'] = line_h
        
        # 3. Straight line (vertical)
        line_v = torch.zeros(1, size, size, dtype=torch.bool)
        line_v[0, :, size//2] = True
        patterns['vertical_line'] = line_v
        
        # 4. Snake pattern (worst case for flood fill)
        snake = torch.zeros(1, size, size, dtype=torch.bool)
        for i in range(size):
            if i % 2 == 0:
                snake[0, i, 0:size-1] = True
            else:
                snake[0, i, 1:size] = True
        # Connect the rows
        for i in range(size-1):
            if i % 2 == 0:
                snake[0, i, size-2] = True
            else:
                snake[0, i, 1] = True
        patterns['snake_pattern'] = snake
        
        # 5. Solid square (different sizes)
        for square_size in [3, 5, 9, 13]:
            square = torch.zeros(1, size, size, dtype=torch.bool)
            start = (size - square_size) // 2
            square[0, start:start+square_size, start:start+square_size] = True
            patterns[f'square_{square_size}x{square_size}'] = square
        
        # 6. Spiral pattern
        spiral = self._create_spiral_pattern(size)
        patterns['spiral'] = spiral
        
        # 7. Random clusters
        torch.manual_seed(42)
        for density in [0.1, 0.3, 0.5]:
            random = torch.rand(1, size, size) < density
            patterns[f'random_{int(density*100)}%'] = random
        
        # 8. Cross pattern
        cross = torch.zeros(1, size, size, dtype=torch.bool)
        cross[0, size//2, :] = True
        cross[0, :, size//2] = True
        patterns['cross'] = cross
        
        # 9. Diagonal line (not connected in Go rules!)
        diagonal = torch.zeros(1, size, size, dtype=torch.bool)
        for i in range(size):
            diagonal[0, i, i] = True
        patterns['diagonal_disconnected'] = diagonal
        
        # 10. Frame (hollow square)
        frame = torch.zeros(1, size, size, dtype=torch.bool)
        frame[0, 0, :] = True
        frame[0, -1, :] = True
        frame[0, :, 0] = True
        frame[0, :, -1] = True
        patterns['frame'] = frame
        
        return patterns
    
    def _create_spiral_pattern(self, size: int) -> torch.Tensor:
        """Create a spiral pattern"""
        spiral = torch.zeros(1, size, size, dtype=torch.bool)
        
        # Start from center
        x, y = size // 2, size // 2
        spiral[0, x, y] = True
        
        # Spiral outward
        dx, dy = 0, 1
        steps = 1
        
        for _ in range(size * 2):
            for _ in range(2):
                for _ in range(steps):
                    x, y = x + dx, y + dy
                    if 0 <= x < size and 0 <= y < size:
                        spiral[0, x, y] = True
                dx, dy = -dy, dx
            steps += 1
            
        return spiral
    
    def analyze_all_patterns(self) -> List[Dict]:
        """Analyze iteration counts for all patterns"""
        patterns = self.create_test_patterns()
        results = []
        
        for name, pattern in patterns.items():
            # Use center as seed for most patterns
            if 'random' in name:
                # For random, find first True position
                true_positions = pattern.nonzero(as_tuple=False)
                if len(true_positions) > 0:
                    seed_pos = true_positions[0]
                    seeds = torch.zeros_like(pattern)
                    seeds[tuple(seed_pos)] = True
                else:
                    continue
            else:
                # Use top-left of pattern as seed
                true_positions = pattern.nonzero(as_tuple=False)
                if len(true_positions) > 0:
                    seed_pos = true_positions[0]
                    seeds = torch.zeros_like(pattern)
                    seeds[tuple(seed_pos)] = True
                else:
                    continue
            
            # Run flood fill
            result, iterations = self._flood_fill_with_count(seeds, pattern)
            
            # Calculate statistics
            total_stones = pattern.sum().item()
            connected_stones = result.sum().item()
            
            results.append({
                'pattern': name,
                'iterations': iterations,
                'total_stones': total_stones,
                'connected_stones': connected_stones,
                'stones_per_iteration': connected_stones / iterations if iterations > 0 else 0
            })
        
        return results
    
    def analyze_board_size_scaling(self) -> Dict[int, List[Tuple[str, int]]]:
        """Analyze how iteration count scales with board size"""
        board_sizes = [5, 9, 13, 19, 25]
        scaling_results = {}
        
        for board_size in board_sizes:
            analyzer = FloodFillAnalyzer(board_size)
            
            # Test specific patterns
            results = []
            
            # Worst case: snake that covers most of board
            snake = torch.zeros(1, board_size, board_size, dtype=torch.bool)
            for i in range(board_size):
                if i % 2 == 0:
                    snake[0, i, :] = True
                else:
                    snake[0, i, :] = True
            seeds = torch.zeros_like(snake)
            seeds[0, 0, 0] = True
            _, iterations = analyzer._flood_fill_with_count(seeds, snake)
            results.append(('snake', iterations))
            
            # Line across board
            line = torch.zeros(1, board_size, board_size, dtype=torch.bool)
            line[0, board_size//2, :] = True
            seeds = torch.zeros_like(line)
            seeds[0, board_size//2, 0] = True
            _, iterations = analyzer._flood_fill_with_count(seeds, line)
            results.append(('line', iterations))
            
            # Square
            square = torch.ones(1, board_size, board_size, dtype=torch.bool)
            seeds = torch.zeros_like(square)
            seeds[0, 0, 0] = True
            _, iterations = analyzer._flood_fill_with_count(seeds, square)
            results.append(('full_board', iterations))
            
            scaling_results[board_size] = results
        
        return scaling_results
    
    def visualize_iteration_spread(self):
        """Visualize how flood fill spreads iteration by iteration"""
        # Create a simple L-shaped pattern for visualization
        size = 9
        pattern = torch.zeros(1, size, size, dtype=torch.bool)
        
        # Create L shape
        pattern[0, 1:7, 1] = True  # Vertical part
        pattern[0, 6, 1:6] = True  # Horizontal part
        
        seeds = torch.zeros_like(pattern)
        seeds[0, 1, 1] = True  # Start from top of L
        
        # Track iteration by iteration
        groups = seeds.clone()
        iteration_map = torch.full((size, size), -1, dtype=torch.int)
        iteration_map[1, 1] = 0  # Seed is iteration 0
        
        print("Tracking flood fill spread iteration by iteration:")
        print("Pattern shape: L")
        print(f"Total positions to fill: {pattern.sum().item()}")
        print()
        
        for iteration in range(20):  # More than enough
            # Clear expanded buffer
            expanded = torch.zeros_like(groups)
            
            # Get neighbors
            expanded[:, 1:, :] |= groups[:, :-1, :]
            expanded[:, :-1, :] |= groups[:, 1:, :]
            expanded[:, :, 1:] |= groups[:, :, :-1]
            expanded[:, :, :-1] |= groups[:, :, 1:]
            
            # Apply constraints
            expanded &= pattern
            expanded &= ~groups
            
            if not expanded.any():
                print(f"Converged after {iteration + 1} iterations\n")
                break
            
            # Mark which iteration found these positions
            new_positions = expanded[0].nonzero(as_tuple=False)
            for pos in new_positions:
                iteration_map[pos[0], pos[1]] = iteration + 1
            
            # Update groups
            groups |= expanded
            
            print(f"Iteration {iteration + 1}: Found {expanded.sum().item()} new positions")
        
        # Visualize the iteration map
        print("Iteration map (when each position was discovered):")
        print("  ", end="")
        for j in range(size):
            print(f"{j:2}", end=" ")
        print()
        
        for i in range(size):
            print(f"{i}: ", end="")
            for j in range(size):
                val = iteration_map[i, j].item()
                if val == -1:
                    print(" .", end=" ")
                else:
                    print(f"{val:2}", end=" ")
            print()
        
        return iteration_map


def main():
    """Run the analysis and display results"""
    print("="*60)
    print("FLOOD FILL ITERATION COUNT ANALYSIS")
    print("="*60)
    
    analyzer = FloodFillAnalyzer(19)  # Standard Go board
    
    # Analyze all patterns
    print("\n1. Iteration counts for various patterns (19x19 board):")
    print("-"*60)
    results = analyzer.analyze_all_patterns()
    
    # Sort by iteration count
    results.sort(key=lambda x: x['iterations'])
    
    print(f"{'Pattern':<25} {'Iterations':>10} {'Stones':>10} {'Connected':>10} {'Stones/Iter':>12}")
    print("-"*60)
    for r in results:
        print(f"{r['pattern']:<25} {r['iterations']:>10} {r['total_stones']:>10} "
              f"{r['connected_stones']:>10} {r['stones_per_iteration']:>12.2f}")
    
    # Analyze scaling
    print("\n2. How iterations scale with board size:")
    print("-"*60)
    scaling = analyzer.analyze_board_size_scaling()
    
    print(f"{'Board Size':<12}", end="")
    for pattern_name in ['line', 'full_board', 'snake']:
        print(f"{pattern_name:>15}", end="")
    print()
    print("-"*60)
    
    for board_size, results in scaling.items():
        print(f"{board_size:>10}  ", end="")
        pattern_dict = dict(results)
        for pattern in ['line', 'full_board', 'snake']:
            if pattern in pattern_dict:
                print(f"{pattern_dict[pattern]:>15}", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()
    
    # Show theoretical limits
    print("\n3. Theoretical iteration limits:")
    print("-"*60)
    for board_size in [9, 13, 19, 25]:
        max_iterations = min(board_size * board_size // 4, 100)
        print(f"Board size {board_size}x{board_size}: max_flood_iterations = "
              f"min({board_size}²/4, 100) = min({board_size*board_size//4}, 100) = {max_iterations}")
    
    # Visualize spread
    print("\n4. Visualizing iteration-by-iteration spread:")
    print("-"*60)
    small_analyzer = FloodFillAnalyzer(9)
    small_analyzer.visualize_iteration_spread()
    
    # Key insights
    print("\n5. KEY INSIGHTS:")
    print("-"*60)
    print("• Single stone: 1 iteration (just checks neighbors)")
    print("• Straight line: ~board_size iterations (spreads linearly)")
    print("• Solid square: ~sqrt(area) iterations (spreads radially)")
    print("• Snake pattern: Can approach max_iterations (worst case)")
    print("• Most realistic Go patterns: 5-20 iterations")
    print("• The bound prevents infinite loops while allowing complex shapes")


if __name__ == "__main__":
    main()