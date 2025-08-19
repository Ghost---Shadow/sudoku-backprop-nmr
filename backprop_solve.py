import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
        """Vectorized exclusion loss"""
        # Row exclusion loss
        row_sums = torch.sum(grid_probs, dim=1)
        row_loss = torch.sum((row_sums - 1.0) ** 2)

        # Column exclusion loss
        col_sums = torch.sum(grid_probs, dim=0)
        col_loss = torch.sum((col_sums - 1.0) ** 2)

        # Box exclusion loss
        box_probs = grid_probs.view(3, 3, 3, 3, 9)
        box_sums = torch.sum(box_probs, dim=(2, 3))
        box_loss = torch.sum((box_sums - 1.0) ** 2)

        return row_loss + col_loss + box_loss


class SudokuSolver(nn.Module):
    def __init__(self, initial_grid):
        super().__init__()

        # Create the parameter tensor (9x9x9)
        self.grid_logits = nn.Parameter(torch.zeros(9, 9, 9))

        # Store the mask for unknown cells
        grid_tensor = torch.tensor(initial_grid, dtype=torch.long)
        self.register_buffer("unknown_mask", grid_tensor == UNK)

        # Initialize based on the known/unknown cells
        with torch.no_grad():
            self.grid_logits.fill_(0.0)  # Unknown cells start uniform
            known_mask = ~self.unknown_mask
            if known_mask.any():
                known_rows, known_cols = torch.where(known_mask)
                correct_numbers = grid_tensor[known_mask] - 1
                self.grid_logits[known_rows, known_cols, :] = -20.0
                self.grid_logits[known_rows, known_cols, correct_numbers] = 20.0

        # Register hook to mask gradients for known cells
        self.grid_logits.register_hook(self._mask_gradients)

    def _mask_gradients(self, grad):
        """Mask gradients for known cells"""
        mask_expanded = self.unknown_mask.unsqueeze(-1).expand_as(grad)
        return grad * mask_expanded.float()

    def forward(self):
        grid_probs = torch.softmax(self.grid_logits, dim=-1)
        return grid_probs

    def get_solution(self):
        with torch.no_grad():
            grid_probs = self.forward()
            solution = torch.argmax(grid_probs, dim=-1) + 1
            return solution.numpy()

    def get_fid_signal(self):
        """Calculate the FID signal - coherence of unknown cells"""
        with torch.no_grad():
            grid_probs = self.forward()

            # Extract probabilities only for unknown cells
            unknown_probs = grid_probs[self.unknown_mask]  # Shape: (num_unknown, 9)

            # Calculate "magnetization" - deviation from equilibrium (uniform)
            equilibrium = torch.ones_like(unknown_probs) / 9.0
            magnetization = unknown_probs - equilibrium

            # FID signal components:
            # 1. Total coherence (sum of absolute magnetization)
            total_coherence = torch.sum(torch.abs(magnetization))

            # 2. Phase coherence (complex-like signal)
            # Use the maximum probability as "real" part and entropy as "imaginary" part
            max_probs = torch.max(unknown_probs, dim=1)[0]
            entropy = -torch.sum(unknown_probs * torch.log(unknown_probs + 1e-8), dim=1)

            # Average over all unknown cells
            real_signal = torch.mean(max_probs - 1 / 9)  # Deviation from uniform
            imag_signal = torch.mean(entropy - np.log(9))  # Deviation from max entropy

            return {
                "total_coherence": total_coherence.item(),
                "real": real_signal.item(),
                "imag": imag_signal.item(),
                "magnitude": (real_signal**2 + imag_signal**2).sqrt().item(),
            }


# Initialize the model
model = SudokuSolver(unsolved_grid)

# Initialize loss functions
bistable_loss = BistableLoss()
exclusion_loss = ExclusionLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# FID recording
num_epochs = 1000
fid_data = {
    "epoch": [],
    "total_coherence": [],
    "real": [],
    "imag": [],
    "magnitude": [],
    "bistable_loss": [],
    "exclusion_loss": [],
    "total_loss": [],
}

solution_found_epoch = None

print("🧲 Starting Sudoku NMR - Recording Free Induction Decay...")
print("📡 Initial RF pulse applied - protons knocked out of equilibrium!")
print()

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    grid_probs = model()

    # Calculate losses
    bistable = bistable_loss(grid_probs)
    exclusion = exclusion_loss(grid_probs)
    total_loss = bistable + 0.1 * exclusion

    # Backward pass
    total_loss.backward()
    optimizer.step()

    # Record FID signal
    fid_signal = model.get_fid_signal()
    fid_data["epoch"].append(epoch)
    fid_data["total_coherence"].append(fid_signal["total_coherence"])
    fid_data["real"].append(fid_signal["real"])
    fid_data["imag"].append(fid_signal["imag"])
    fid_data["magnitude"].append(fid_signal["magnitude"])
    fid_data["bistable_loss"].append(bistable.item())
    fid_data["exclusion_loss"].append(exclusion.item())
    fid_data["total_loss"].append(total_loss.item())

    # Check if solution is valid
    current_solution = model.get_solution()
    if is_valid_sudoku(current_solution) and solution_found_epoch is None:
        solution_found_epoch = epoch
        print(f"🎯 *** SOLUTION FOUND AT EPOCH {epoch}! ***")
        print("📊 NMR signal collapsed to ground state!")
        print("Solution:")
        print(current_solution)
        print()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: Loss = {total_loss.item():.4f}, "
            f"FID Magnitude = {fid_signal['magnitude']:.4f}, "
            f"Coherence = {fid_signal['total_coherence']:.2f}"
        )

print("\n🔬 NMR Experiment Complete - Analyzing Spectral Data...")

# Create the FID and spectral plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: FID Time Domain Signal (Real and Imaginary)
ax1.plot(fid_data["epoch"], fid_data["real"], "b-", label="Real", alpha=0.7)
ax1.plot(fid_data["epoch"], fid_data["imag"], "r-", label="Imaginary", alpha=0.7)
ax1.set_xlabel("Time (Epochs)")
ax1.set_ylabel("Signal Amplitude")
ax1.set_title("Sudoku FID Signal - Time Domain")
ax1.legend()
ax1.grid(True, alpha=0.3)
if solution_found_epoch:
    ax1.axvline(
        solution_found_epoch,
        color="green",
        linestyle="--",
        label=f"Solution Found (t={solution_found_epoch})",
    )
    ax1.legend()

# Plot 2: FID Magnitude and Total Coherence
ax2.plot(
    fid_data["epoch"], fid_data["magnitude"], "purple", label="Magnitude", linewidth=2
)
ax2_twin = ax2.twinx()
ax2_twin.plot(
    fid_data["epoch"],
    fid_data["total_coherence"],
    "orange",
    label="Total Coherence",
    alpha=0.7,
)
ax2.set_xlabel("Time (Epochs)")
ax2.set_ylabel("FID Magnitude", color="purple")
ax2_twin.set_ylabel("Total Coherence", color="orange")
ax2.set_title("Sudoku FID - Magnitude and Coherence")
ax2.grid(True, alpha=0.3)

# Plot 3: Frequency Domain (FFT of FID)
# Create complex signal for FFT
complex_signal = np.array(fid_data["real"]) + 1j * np.array(fid_data["imag"])
fft_signal = np.fft.fft(complex_signal)
frequencies = np.fft.fftfreq(len(fft_signal))

ax3.plot(
    frequencies[: len(frequencies) // 2],
    np.abs(fft_signal)[: len(fft_signal) // 2],
    "green",
)
ax3.set_xlabel("Frequency (cycles/epoch)")
ax3.set_ylabel("Spectral Intensity")
ax3.set_title("Sudoku NMR Spectrum - Frequency Domain")
ax3.grid(True, alpha=0.3)

# Plot 4: Loss Evolution
ax4.semilogy(fid_data["epoch"], fid_data["total_loss"], "black", label="Total Loss")
ax4.semilogy(
    fid_data["epoch"],
    fid_data["bistable_loss"],
    "blue",
    alpha=0.7,
    label="Bistable (B₀ field)",
)
ax4.semilogy(
    fid_data["epoch"],
    fid_data["exclusion_loss"],
    "red",
    alpha=0.7,
    label="Exclusion (Coupling)",
)
ax4.set_xlabel("Time (Epochs)")
ax4.set_ylabel("Loss (Log Scale)")
ax4.set_title("NMR Energy Levels - Loss Evolution")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle("🧲 Sudoku NMR Spectroscopy Analysis 📡", fontsize=16, y=1.02)
plt.savefig("sudoku_nmr_analysis.png", dpi=300, bbox_inches="tight")
plt.close()  # Close the figure to free memory
print("📁 Spectral analysis saved to: sudoku_nmr_analysis.png")

# Final results
print(f"\n🎯 Final Results:")
print(f"Final solution:")
final_solution = model.get_solution()
print(final_solution)
print(f"Is solution valid? {is_valid_sudoku(final_solution)}")

if solution_found_epoch is not None:
    print(f"\n🏆 Solution found at epoch {solution_found_epoch}")
    print(f"📈 Final FID magnitude: {fid_data['magnitude'][-1]:.6f}")
    print(f"🔄 Final coherence: {fid_data['total_coherence'][-1]:.2f}")
else:
    print("\n❌ No valid solution found - signal did not fully relax")

print(f"\n📊 Spectral Characteristics:")
print(
    f"Peak frequency: {frequencies[np.argmax(np.abs(fft_signal)[:len(fft_signal)//2])]:.4f} cycles/epoch"
)
print(f"Dominant spectral intensity: {np.max(np.abs(fft_signal)):.2f}")
print(f"Initial coherence: {fid_data['total_coherence'][0]:.2f}")
print(f"Final coherence: {fid_data['total_coherence'][-1]:.2f}")
print(
    f"Relaxation ratio: {fid_data['total_coherence'][-1]/fid_data['total_coherence'][0]:.4f}"
)
