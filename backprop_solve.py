import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the unsolved grid
UNK = -1
unsolved_grid = np.array(
    [
        [6, 7, UNK, 1, 5, 4, 3, 9, 8],
        [1, UNK, 4, 7, 8, 9, 2, 6, 5],
        [9, 8, 5, 3, 6, 2, UNK, 4, 1],
        [5, UNK, 3, 6, 1, UNK, 9, 8, 4],
        [4, 6, UNK, 9, 2, 8, 5, 1, 3],
        [8, 1, 9, 4, 3, 5, 6, UNK, 7],
        [UNK, 4, UNK, UNK, UNK, 6, UNK, 3, 9],
        [UNK, 9, 6, UNK, 4, 3, 1, 5, UNK],
        [3, 5, UNK, UNK, 9, UNK, 4, 7, 6],
    ]
)


class BistableLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = x**2
        b = (x - 1) ** 2
        return (a * b).mean()


class ExclusionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grid_probs):
        """
        grid_probs: (9, 9, 9) tensor where grid_probs[i,j,k] is probability
        that cell (i,j) contains number (k+1)
        """
        total_loss = 0.0

        # Row exclusion loss - penalize when sum of probabilities > 1
        for i in range(9):  # for each row
            for k in range(9):  # for each number (0-8 representing 1-9)
                row_probs = grid_probs[i, :, k]  # shape: (9,)
                prob_sum = torch.sum(row_probs)
                # Penalize deviation from exactly 1
                total_loss += (prob_sum - 1.0) ** 2

        # Column exclusion loss
        for j in range(9):  # for each column
            for k in range(9):  # for each number
                col_probs = grid_probs[:, j, k]  # shape: (9,)
                prob_sum = torch.sum(col_probs)
                total_loss += (prob_sum - 1.0) ** 2

        # Box exclusion loss
        for box_i in range(3):  # 3x3 boxes
            for box_j in range(3):
                for k in range(9):  # for each number
                    # Extract 3x3 box
                    box_probs = grid_probs[
                        box_i * 3 : (box_i + 1) * 3, box_j * 3 : (box_j + 1) * 3, k
                    ]
                    prob_sum = torch.sum(box_probs)
                    total_loss += (prob_sum - 1.0) ** 2

        return total_loss


class SudokuSolver(nn.Module):
    def __init__(self, initial_grid):
        super().__init__()

        # Create the parameter tensor (9x9x9)
        # Each cell has 9 logits (before softmax) for numbers 1-9
        self.grid_logits = nn.Parameter(torch.zeros(9, 9, 9))

        # Initialize based on the known/unknown cells
        with torch.no_grad():
            for i in range(9):
                for j in range(9):
                    if initial_grid[i, j] != UNK:
                        # Known cell - set high logit for correct number, low for others
                        number = initial_grid[i, j] - 1  # Convert to 0-8 indexing
                        self.grid_logits[i, j, :] = -10.0  # Low probability for all
                        self.grid_logits[i, j, number] = (
                            10.0  # High probability for correct number
                        )
                    else:
                        # Unknown cell - initialize with equal logits (uniform after softmax)
                        self.grid_logits[i, j, :] = 0.0

    def forward(self):
        # Convert logits to probabilities using softmax
        # Each cell becomes a probability distribution over numbers 1-9
        grid_probs = torch.softmax(self.grid_logits, dim=-1)
        return grid_probs

    def get_solution(self):
        """Extract the most likely solution"""
        with torch.no_grad():
            grid_probs = self.forward()
            solution = torch.argmax(grid_probs, dim=-1) + 1  # Convert back to 1-9
            return solution.numpy()


# Initialize the model
model = SudokuSolver(unsolved_grid)

# Initialize loss functions
bistable_loss = BistableLoss()
exclusion_loss = ExclusionLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    grid_probs = model()

    # Calculate losses
    bistable = bistable_loss(grid_probs)
    exclusion = exclusion_loss(grid_probs)

    # Total loss (you can adjust the weights)
    total_loss = bistable + 0.1 * exclusion

    # Backward pass
    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: Total Loss = {total_loss.item():.4f}, "
            f"Bistable = {bistable.item():.4f}, Exclusion = {exclusion.item():.4f}"
        )

        # Check current solution
        current_solution = model.get_solution()
        print("Current solution:")
        print(current_solution)
        print()

# Final solution
print("Final solution:")
final_solution = model.get_solution()
print(final_solution)


# Verify the solution
def is_valid_sudoku(grid):
    def is_valid_group(group):
        group = [x for x in group if x != 0]
        return len(group) == len(set(group))

    # Check rows
    for row in grid:
        if not is_valid_group(row):
            return False

    # Check columns
    for col in range(9):
        if not is_valid_group([grid[row][col] for row in range(9)]):
            return False

    # Check 3x3 boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = [grid[r][c] for r in range(i, i + 3) for c in range(j, j + 3)]
            if not is_valid_group(box):
                return False

    return True


print(f"\nIs solution valid? {is_valid_sudoku(final_solution)}")

# Show probability distribution for a few unknown cells
print("\nProbability distributions for some unknown cells:")
with torch.no_grad():
    final_probs = model()
    for i in range(9):
        for j in range(9):
            if unsolved_grid[i, j] == UNK:
                probs = final_probs[i, j].numpy()
                max_prob_idx = np.argmax(probs)
                print(
                    f"Cell ({i},{j}): Number {max_prob_idx + 1} with probability {probs[max_prob_idx]:.3f}"
                )
                break
        if unsolved_grid[i, j] == UNK:
            break
