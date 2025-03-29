import numpy as np
from typing import List, Tuple

def calculate_row_major(num_programs: int, num_pid_n: int) -> List[Tuple[int, int]]:
    """
    Calculate a row-major mapping for program IDs.

    Each program ID is mapped to a tuple (row, col) using:
      - row = pid // num_pid_n
      - col = pid % num_pid_n

    Args:
        num_programs (int): Total number of programs.
        num_pid_n (int): Number of program IDs per row.

    Returns:
        List[Tuple[int, int]]: A list of (row, col) tuples for each program ID.
    """

    return [(pid // num_pid_n, pid % num_pid_n) for pid in range(num_programs)]


def create_grid(mapping: List[Tuple[int, int]], num_rows: int, num_cols: int) -> np.ndarray:
    """
    Create a 2D grid populated with program IDs based on the provided mapping.

    The grid is initially filled with -1, and then each (row, col) in the mapping
    is updated with the corresponding program ID.

    Args:
        mapping (List[Tuple[int, int]]): List of (row, col) tuples mapping program IDs.
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.

    Returns:
        np.ndarray: A 2D grid with program IDs placed at their mapped coordinates.
    """

    grid = np.full((num_rows, num_cols), -1, dtype=int)
    for pid, (m, n) in enumerate(mapping):
        grid[m, n] = pid
    return grid


def print_grid(title: str, grid: np.ndarray, GROUP_SIZE_M: int) -> None:
    """
    Print the grid with a title and formatted borders.

    This function displays the grid with each cell formatted to a fixed width.
    Horizontal separators are added every GROUP_SIZE_M rows.

    Args:
        title (str): Title to display above the grid.
        grid (np.ndarray): The 2D grid to print.
        GROUP_SIZE_M (int): The number of rows in each group before adding a separator.
    """

    print(f"\n{title}:")
    print("+" + "-" * (6 * grid.shape[1] - 1) + "+")
    for row_idx, row in enumerate(grid):
        cells = [f"{val:3d}" for val in row]
        print("| " + " | ".join(cells) + " |")

        if (row_idx + 1) % GROUP_SIZE_M == 0 and row_idx != len(grid) - 1:
            print("+" + "-" * (6 * grid.shape[1] - 1) + "+")

    print("+" + "-" * (6 * grid.shape[1] - 1) + "+")


def visualize_matrix_mapping(M: int, N: int, BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, grid: Tuple[int, int]) -> None:
    """
    Visualize how a matrix is partitioned into blocks (tiles).

    This function divides an MxN matrix into blocks of size BLOCK_SIZE_M x BLOCK_SIZE_N.
    Each block is assigned a unique linear block ID based on its position in a grid.
    The resulting mapping is printed with vertical and horizontal separators.

    Args:
        M (int): Total number of rows in the matrix.
        N (int): Total number of columns in the matrix.
        BLOCK_SIZE_M (int): Number of rows per block.
        BLOCK_SIZE_N (int): Number of columns per block.
        grid (tuple): A tuple (num_blocks_m, num_blocks_n) indicating the number of blocks along M and N.
    """

    num_blocks_m, num_blocks_n = grid[0], grid[1]
    # Initialize an M x N grid with -1.
    grid = np.full((M, N), -1, dtype=int)

    # Fill grid with block IDs
    for block_m in range(num_blocks_m):
        for block_n in range(num_blocks_n):
            # Calculate the boundaries of the current block.
            m_start = block_m * BLOCK_SIZE_M
            n_start = block_n * BLOCK_SIZE_N
            m_end = min(m_start + BLOCK_SIZE_M, M)
            n_end = min(n_start + BLOCK_SIZE_N, N)
            # Calculate linear block ID
            block_id = block_m * num_blocks_n + block_n
            # Fill the corresponding block region in the grid with the block ID.
            grid[m_start:m_end, n_start:n_end] = block_id

    print("C Matrix Block Mapping Visualization (20x16 matrix):")
    print(f"Each number represents a {num_blocks_m}x{num_blocks_n} block ID \n")

    for m in range(M):
        row = [f"{grid[m, n]:3d}" if grid[m, n] != -1 else "   " for n in range(N)]
        # Insert vertical separators every BLOCK_SIZE_N
        formatted_row = "|".join(
            ["".join(row[i : i + BLOCK_SIZE_N]) for i in range(0, N, BLOCK_SIZE_N)]
        )
        print(f"|{formatted_row}|")
        # Insert horizontal separators every BLOCK_SIZE_M
        if (m + 1) % BLOCK_SIZE_M == 0 and m != M - 1:
            print("+" + "-" * (len(formatted_row)) + "+")
    print("+" + "-" * (len(formatted_row)) + "+")
