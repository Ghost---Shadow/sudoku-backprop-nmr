import numpy as np

# UNSOLVED VERSION - Original puzzle with UNK tokens for handwritten entries
UNK = -1  # Token for handwritten/uncertain entries

# Based on careful analysis of the image - printed vs handwritten numbers
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


def solve_sudoku(grid):
    """
    Solve a Sudoku puzzle using backtracking.
    Input grid should have 0 for empty cells.
    Returns solved grid or None if unsolvable.
    """

    def is_valid(grid, row, col, num):
        # Check row
        for x in range(9):
            if grid[row][x] == num:
                return False

        # Check column
        for x in range(9):
            if grid[x][col] == num:
                return False

        # Check 3x3 box
        start_row = row - row % 3
        start_col = col - col % 3
        for i in range(3):
            for j in range(3):
                if grid[i + start_row][j + start_col] == num:
                    return False
        return True

    def solve(grid):
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(grid, i, j, num):
                            grid[i][j] = num
                            if solve(grid):
                                return True
                            grid[i][j] = 0
                    return False
        return True

    # Create a copy to avoid modifying original
    work_grid = grid.copy()
    if solve(work_grid):
        return work_grid
    return None


# Convert unsolved grid to format for solver (UNK -> 0)
unsolved_for_solver = np.where(unsolved_grid == UNK, 0, unsolved_grid)

print("Attempting to solve the puzzle automatically...")
auto_solved = solve_sudoku(unsolved_for_solver)

if auto_solved is not None:
    print("✓ Puzzle solved successfully!")
    solved_grid = auto_solved
    print("\nCORRECT SOLVED GRID:")
    print(solved_grid)
else:
    print("✗ Could not solve puzzle - there may be errors in the unsolved grid")
    # Keep the manually transcribed version for comparison
    solved_grid = np.array(
        [
            [6, 7, 2, 1, 5, 4, 3, 9, 8],
            [1, 3, 4, 7, 8, 9, 2, 6, 5],
            [9, 8, 5, 3, 6, 2, 7, 4, 1],
            [5, 2, 3, 6, 1, 7, 9, 8, 4],
            [4, 6, 7, 9, 2, 8, 5, 1, 3],
            [8, 1, 9, 4, 3, 5, 6, 2, 7],
            [2, 4, 8, 5, 7, 6, 1, 3, 9],
            [7, 9, 6, 8, 4, 3, 8, 5, 2],
            [3, 5, 1, 2, 9, 8, 4, 7, 6],
        ]
    )

print("UNSOLVED GRID (with UNK tokens):")
print(unsolved_grid)
print("\nSOLVED GRID (all numbers filled):")
print(solved_grid)


# VERIFICATION FUNCTIONS
def verify_sudoku_grid(grid, grid_name="Grid"):
    """
    Verify a Sudoku grid for common mistakes.
    Returns a dictionary with verification results.
    """
    results = {
        "grid_name": grid_name,
        "valid_numbers": True,
        "row_duplicates": [],
        "col_duplicates": [],
        "box_duplicates": [],
        "impossible_values": [],
        "is_complete": True,
    }

    # For unsolved grids, replace UNK with 0 for checking
    if np.any(grid == UNK):
        check_grid = np.where(grid == UNK, 0, grid)
        results["is_complete"] = False
    else:
        check_grid = grid.copy()

    # Check for valid numbers (1-9, 0, or UNK)
    valid_mask = ((grid >= 1) & (grid <= 9)) | (grid == 0) | (grid == UNK)
    if not np.all(valid_mask):
        results["valid_numbers"] = False
        invalid_pos = np.where(~valid_mask)
        results["impossible_values"] = [
            (
                invalid_pos[0][i],
                invalid_pos[1][i],
                grid[invalid_pos[0][i], invalid_pos[1][i]],
            )
            for i in range(len(invalid_pos[0]))
        ]

    # Check rows for duplicates (excluding 0s)
    for i in range(9):
        row = check_grid[i]
        non_zero = row[row != 0]
        if len(non_zero) != len(np.unique(non_zero)):
            unique, counts = np.unique(non_zero, return_counts=True)
            duplicates = unique[counts > 1]
            results["row_duplicates"].append((i, duplicates.tolist()))

    # Check columns for duplicates (excluding 0s)
    for j in range(9):
        col = check_grid[:, j]
        non_zero = col[col != 0]
        if len(non_zero) != len(np.unique(non_zero)):
            unique, counts = np.unique(non_zero, return_counts=True)
            duplicates = unique[counts > 1]
            results["col_duplicates"].append((j, duplicates.tolist()))

    # Check 3x3 boxes for duplicates (excluding 0s)
    for box_row in range(3):
        for box_col in range(3):
            box = check_grid[
                box_row * 3 : (box_row + 1) * 3, box_col * 3 : (box_col + 1) * 3
            ]
            box_flat = box.flatten()
            non_zero = box_flat[box_flat != 0]
            if len(non_zero) != len(np.unique(non_zero)):
                unique, counts = np.unique(non_zero, return_counts=True)
                duplicates = unique[counts > 1]
                results["box_duplicates"].append(
                    ((box_row, box_col), duplicates.tolist())
                )

    return results


def print_verification_results(results):
    """Print verification results in a readable format."""
    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULTS FOR {results['grid_name'].upper()}")
    print(f"{'='*60}")

    if results["valid_numbers"]:
        print("✓ All numbers are valid")
    else:
        print("✗ Invalid numbers found:")
        for row, col, val in results["impossible_values"]:
            print(f"  Position ({row}, {col}): {val}")

    if not results["row_duplicates"]:
        print("✓ No duplicate numbers in any row")
    else:
        print("✗ Duplicate numbers found in rows:")
        for row, duplicates in results["row_duplicates"]:
            print(f"  Row {row}: duplicates {duplicates}")
            # Show the actual row for debugging
            grid_to_check = (
                solved_grid
                if "SOLVED" in results["grid_name"].upper()
                else unsolved_grid
            )
            print(f"    Row content: {grid_to_check[row]}")

    if not results["col_duplicates"]:
        print("✓ No duplicate numbers in any column")
    else:
        print("✗ Duplicate numbers found in columns:")
        for col, duplicates in results["col_duplicates"]:
            print(f"  Column {col}: duplicates {duplicates}")
            # Show the actual column for debugging
            grid_to_check = (
                solved_grid
                if "SOLVED" in results["grid_name"].upper()
                else unsolved_grid
            )
            print(f"    Column content: {grid_to_check[:, col]}")

    if not results["box_duplicates"]:
        print("✓ No duplicate numbers in any 3x3 box")
    else:
        print("✗ Duplicate numbers found in 3x3 boxes:")
        for (box_row, box_col), duplicates in results["box_duplicates"]:
            print(f"  Box ({box_row}, {box_col}): duplicates {duplicates}")
            # Show the actual box for debugging
            grid_to_check = (
                solved_grid
                if "SOLVED" in results["grid_name"].upper()
                else unsolved_grid
            )
            box_content = grid_to_check[
                box_row * 3 : (box_row + 1) * 3, box_col * 3 : (box_col + 1) * 3
            ]
            print(f"    Box content:\n{box_content}")

    # Overall assessment
    all_good = (
        results["valid_numbers"]
        and not results["row_duplicates"]
        and not results["col_duplicates"]
        and not results["box_duplicates"]
    )

    print(f"\n{'-'*60}")
    if all_good:
        status = (
            "COMPLETE AND VALID" if results["is_complete"] else "VALID (INCOMPLETE)"
        )
        print(f"✓ OVERALL: {status}")
        print("  No transcription errors detected!")
    else:
        print("✗ OVERALL: Issues detected - please review above.")
    print(f"{'-'*60}")


def compare_grids(unsolved, solved):
    """Compare unsolved and solved grids to check consistency."""
    print(f"\n{'='*60}")
    print("CONSISTENCY CHECK: UNSOLVED vs SOLVED")
    print(f"{'='*60}")

    consistent = True
    differences = []

    for i in range(9):
        for j in range(9):
            if unsolved[i, j] != UNK:  # If not a handwritten entry
                if unsolved[i, j] != solved[i, j]:
                    consistent = False
                    differences.append((i, j, unsolved[i, j], solved[i, j]))

    if consistent:
        print("✓ Solved grid is consistent with unsolved grid")
        print("  All original (printed) numbers match between versions")
    else:
        print("✗ Inconsistency found between unsolved and solved grids:")
        for row, col, unsolved_val, solved_val in differences:
            print(
                f"  Position ({row}, {col}): unsolved={unsolved_val}, solved={solved_val}"
            )

    # Show what was filled in
    filled_positions = np.where(unsolved == UNK)
    print(f"\nHandwritten entries filled in ({len(filled_positions[0])} total):")
    for i in range(len(filled_positions[0])):
        row, col = filled_positions[0][i], filled_positions[1][i]
        filled_value = solved[row, col]
        print(f"  Position ({row}, {col}): {filled_value}")


# Run comprehensive verification
print("Starting comprehensive verification...")

# Verify unsolved grid
unsolved_results = verify_sudoku_grid(unsolved_grid, "Unsolved Grid")
print_verification_results(unsolved_results)

# Verify solved grid
solved_results = verify_sudoku_grid(solved_grid, "Solved Grid")
print_verification_results(solved_results)

# Compare consistency
compare_grids(unsolved_grid, solved_grid)

# Final summary
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
handwritten_count = len(np.where(unsolved_grid == UNK)[0])
print(f"Total handwritten entries: {handwritten_count}")
print(
    f"Unsolved grid valid: {'✓' if not (unsolved_results['row_duplicates'] or unsolved_results['col_duplicates'] or unsolved_results['box_duplicates']) else '✗'}"
)
print(
    f"Solved grid valid: {'✓' if not (solved_results['row_duplicates'] or solved_results['col_duplicates'] or solved_results['box_duplicates']) else '✗'}"
)
