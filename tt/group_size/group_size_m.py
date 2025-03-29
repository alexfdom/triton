# %%
import triton
import triton.language as tl
import torch
import numpy as np
import math
from group_utils import calculate_row_major, create_grid, print_grid, visualize_matrix_mapping

# --- Dimension Setup ----- #
M, N = 20, 16                     # C(M, N) = 20x16
BLOCK_SIZE_M, BLOCK_SIZE_N = 4, 4 # 16x16 block size. Grid size = 5x4
GROUP_SIZE_M = 2                  # Group size in M dimension
# ----------------------- #

@triton.jit
def reordered_mapping_kernel(
    pid_ptr, pid_m_ptr, pid_n_ptr,
    M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_triton = tl.program_id(0)
    # Vectorized Approach: Each Triton program processes BLOCK_SIZE coordinates via vectorized indices
    idx = pid_triton * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # Vectorized Approach
    # tl.device_print("idx=", idx)
    mask = idx < NUM_PROGRAMS
    # tl.device_print("idx=", mask)
    
    # Calculate program grid dimensions
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Group calculation
    group_id = idx // num_pid_in_group # idx as a flattened program id
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Reordered coordinates calculation
    pid_m = first_pid_m + ((idx % num_pid_in_group) % group_size_m)
    pid_n = (idx % num_pid_in_group) // group_size_m
    
    # Store results
    tl.store(pid_ptr + idx, idx, mask=mask)
    tl.store(pid_m_ptr + idx, pid_m, mask=mask)
    tl.store(pid_n_ptr + idx, pid_n, mask=mask)

# --- Wrapper function to set up the kernel call --- #
def main():
    print(f"Mapping C matrix ({M}x{N}) to grid {math.ceil(M/BLOCK_SIZE_M)}x{math.ceil(N/BLOCK_SIZE_N)} blocks with GROUP_SIZE_M={GROUP_SIZE_M}")
    # Calculate grid dimensions
    grid = (math.ceil(M/BLOCK_SIZE_M),math.ceil(N/BLOCK_SIZE_N))
    num_pid_m, num_pid_n = grid[0], grid[1]
    num_programs = num_pid_m * num_pid_n

    visualize_matrix_mapping(M,N,BLOCK_SIZE_M,BLOCK_SIZE_N,grid)

    # Generate row-major mapping
    row_major = calculate_row_major(num_programs, num_pid_n)
    row_major_pid = np.arange(num_programs)
    row_major_m = [m for m, _ in row_major]
    row_major_n = [n for _, n in row_major]

    # Generate reordered mapping using Triton kernel
    pid = torch.empty(num_programs, dtype=torch.int32, device='cuda')
    pid_m = torch.empty(num_programs, dtype=torch.int32, device='cuda')
    pid_n = torch.empty(num_programs, dtype=torch.int32, device='cuda')

    # To gain a deeper understanding of the grid variable in a kernel, we will process our flattened pid_triton ID in chunks of BLOCK_SIZE
    BLOCK_SIZE = 8
    grid_kernel = (triton.cdiv(num_programs, BLOCK_SIZE),)
    
    # --- Launch the kernel with the computed grid and kernel arguments --- #
    reordered_mapping_kernel[grid_kernel](
        pid, pid_m, pid_n,
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M,
        num_programs, BLOCK_SIZE
    )
    
    # Convert to numpy arrays
    reordered_pid = pid.cpu().numpy()
    reordered_m = pid_m.cpu().numpy()
    reordered_n = pid_n.cpu().numpy()

    # Create grids
    grid_row_major = create_grid(row_major, num_pid_m, num_pid_n)
    grid_reordered = create_grid(list(zip(reordered_m, reordered_n)), num_pid_m, num_pid_n)

    # Print outputs
    print_grid("Initial Row-Major Mapping", grid_row_major, GROUP_SIZE_M)
    print("pid   :", row_major_pid)
    print("pid_m :", row_major_m)
    print("pid_n :", row_major_n)
    print_grid("Reordered Group Mapping (GROUP_SIZE_M=2)", grid_reordered, GROUP_SIZE_M)
    print("pid   :", reordered_pid)
    print("pid_m :", reordered_m)
    print("pid_n :", reordered_n)


if __name__ == "__main__":
    main()
# %%
