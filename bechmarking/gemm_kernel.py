# %%
import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'

def init_to_zero(*names):
    def result(nargs):
        for name in names:
            nargs[name].zero_()

    return result

def gen_autotune_config(
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    num_stages: int,
    num_warps: int,
    group_size_m: int = 8,
    waves_per_eu: int = None,
) -> triton.Config:
    
    config_dict = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
    }
    
    if waves_per_eu is not None:
        config_dict["waves_per_eu"] = waves_per_eu

    return triton.Config(config_dict, num_stages=num_stages, num_warps=num_warps)


def get_cuda_autotune_config():
    return [
        gen_autotune_config(128, 256, 64, num_stages=3, num_warps=8),
        gen_autotune_config(64, 256, 32, num_stages=4, num_warps=4),
        gen_autotune_config(128, 128, 32, num_stages=4, num_warps=4),
        gen_autotune_config(128, 64, 32, num_stages=4, num_warps=4),
        gen_autotune_config(64, 128, 32, num_stages=4, num_warps=4),
        gen_autotune_config(128, 32, 32, num_stages=4, num_warps=4),
        gen_autotune_config(64, 32, 32, num_stages=5, num_warps=2),
        gen_autotune_config(32, 64, 32, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        gen_autotune_config(128, 256, 128, num_stages=3, num_warps=8),
        gen_autotune_config(256, 128, 128, num_stages=3, num_warps=8),
        gen_autotune_config(256, 64, 128, num_stages=4, num_warps=4),
        gen_autotune_config(64, 256, 128, num_stages=4, num_warps=4),
        gen_autotune_config(128, 128, 128, num_stages=4, num_warps=4),
        gen_autotune_config(128, 64, 64, num_stages=4, num_warps=4),
        gen_autotune_config(64, 128, 64, num_stages=4, num_warps=4),
        gen_autotune_config(128, 32, 64, num_stages=4, num_warps=4),
    ]



def get_hip_autotune_config():
    return [
        gen_autotune_config(128, 256, 16, num_stages=2, num_warps=4, group_size_m=1, waves_per_eu=2),
        gen_autotune_config(256, 256, 16, num_stages=2, num_warps=8, group_size_m=4, waves_per_eu=2),
        gen_autotune_config(128, 128, 32, num_stages=2, num_warps=8, group_size_m=1, waves_per_eu=2),
        gen_autotune_config(64, 128, 32, num_stages=2, num_warps=4, group_size_m=8, waves_per_eu=3),
        gen_autotune_config(64, 64, 32, num_stages=2, num_warps=4, group_size_m=1, waves_per_eu=8),
    ]



def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # GEMM scalars
        alpha: tl.constexpr, beta: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the GEMM C = alpha * (A x B) + beta * C.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        A_tile = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        B_tile = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(A_tile, B_tile, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    C_tile = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    # Load the existing tile from C and scale by beta.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if beta != 0.0:
        C_tile_orig = tl.load(c_ptrs, mask=c_mask, other=0.0)
        # GEMM: Combine the computed partial result with the original C.
        C_tile = alpha * C_tile + beta * C_tile_orig
    else:
        # Skip loading C when beta=0
        C_tile = alpha * C_tile
    
    tl.store(c_ptrs, C_tile, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def gemm(a, b, c, alpha, beta, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    if c.shape != (M, N):
        c = torch.randn((M, N), device=a.device, dtype=torch.float16)
    elif beta != 0:
        c = c.clone()
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    gemm_kernel[grid](
        a, b, c,  
        M, N, K,  
        a.stride(0), a.stride(1),  
        b.stride(0), b.stride(1),  
        c.stride(0), c.stride(1),  
        alpha, beta,  
        ACTIVATION=activation  
    )
    return c

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'
TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2') and hasattr(torch, 'float8_e4m3fn')

alpha_beta_configs = [
    (1.0, 0.0),    
    (1.0, 1.0),    
    (2.0, 0.5),  
]

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    for alpha, beta in alpha_beta_configs:
        configs.append(
            triton.testing.Benchmark(
                x_names=["M", "N", "K"],
                x_vals=[128 * i for i in range(2, 33)],
                line_arg="provider",
                line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],
                line_names=[f"Triton α={alpha}, β={beta}"] if fp8_inputs else 
                          [f"{ref_lib} α={alpha}, β={beta}", f"Triton α={alpha}, β={beta}"],
                styles=[("green", "-"), ("blue", "-")],
                ylabel="TFLOPS",
                plot_name=f"gemm-perf-{'fp8' if fp8_inputs else 'fp16'}-a{alpha}-b{beta}",
                args={"fp8_inputs": fp8_inputs, "alpha": alpha, "beta": beta},
            ))

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs, alpha, beta):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c_initial = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    
    if fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.to(torch.float8_e5m2)

    quantiles = [0.5, 0.2, 0.8]
    
    if provider == ref_lib.lower():
        if fp8_inputs:
            return 0, 0, 0  
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.addmm(c_initial, a, b, beta=beta, alpha=alpha), 
            quantiles=quantiles
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gemm(a, b, c_initial, alpha, beta), 
            quantiles=quantiles
        )

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True)
# %%
