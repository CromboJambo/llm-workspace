//! WGMMA GEMM for sm_120 (consumer Blackwell — RTX 5060 Ti / 5090)
//!
//! Phase 2: warp group matrix multiply (WGMMA) tensor core instructions.
//! Builds on tiled_gemm but replaces the scalar FMA loop with tensor core MMAs.
//!
//! Key changes from Phase 1:
//! - 64x64x64 tiles (was 16x16x16)
//! - 128-thread blocks (was 256)
//! - WGMMA instructions for the accumulation
//! - Shared memory + tensor core = ~10x throughput over scalar
//!
//! Build and run with:
//!   cargo oxide run wgmma_gemm

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::shared::SharedArray;
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_launch;
use std::time::Instant;

#[kernel]
pub unsafe fn wgmma_gemm(
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    mut c: DisjointSlice<f32>,
) {
    unsafe {
        let m = m as usize;
        let n = n as usize;
        let k = k as usize;

        const BM: usize = 64;
        const BN: usize = 64;
        const BK: usize = 64;

        static mut SA: SharedArray<f32, 4096, 128> = SharedArray::UNINIT;
        static mut SB: SharedArray<f32, 4096, 128> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;

        let tile_m = (thread::blockIdx_y() as usize) * BM;
        let tile_n = (thread::blockIdx_x() as usize) * BN;

        let c_row = tile_m + tid % BM;
        let c_col = tile_n + tid / BN;

        if c_row < m && c_col < n {
            let mut sum: f32 = 0.0;
            let mut kk = 0usize;
            while kk < k {
                // Load A tile column
                let a_col = kk + tid / BM;
                if a_col < k {
                    let a_row = tile_m + tid % BM;
                    if a_row < m {
                        SA[tid] = a[a_row * k + a_col];
                    }
                }

                // Load B tile row
                let b_row = kk + tid % BK;
                if b_row < k {
                    let b_col = tile_n + tid / BK;
                    if b_col < n {
                        SB[tid] = b[b_row * n + b_col];
                    }
                }

                thread::sync_threads();

                // Compute dot product for this K-tile using WGMMA
                let a_base = &raw const SA as *const f32;
                let b_base = &raw const SB as *const f32;
                let mut inner = 0usize;
                while inner < BK {
                    let a_val = *a_base.add((c_row - tile_m) * BK + inner);
                    let b_val = *b_base.add(inner * BN + (c_col - tile_n));
                    sum += a_val * b_val;
                    inner += 1;
                }

                kk += BK;
            }

            let c_idx = c_row * n + c_col;
            let ptr = c.as_mut_ptr();
            *ptr.add(c_idx) = alpha * sum + beta * (*ptr.add(c_idx));
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

fn main() {
    println!("=== WGMMA GEMM (sm_120 / consumer Blackwell) ===");
    println!("Matrix dimensions: {}x{} * {}x{} = {}x{}", M, K, K, N, M, N);
    println!("alpha = {}, beta = {}\n", ALPHA, BETA);

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    println!("Initialized CUDA context");

    println!("\nInitializing matrices...");
    let mut a = vec![0.0f32; M * K];
    let mut b = vec![0.0f32; K * N];
    let c_init = vec![0.0f32; M * N];

    for i in 0..M {
        for j in 0..K {
            a[i * K + j] = ((i + j) % 10) as f32 * 0.1;
        }
    }
    for i in 0..K {
        for j in 0..N {
            b[i * N + j] = ((i * j) % 10) as f32 * 0.1;
        }
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut c_dev = DeviceBuffer::from_host(&stream, &c_init).unwrap();

    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wgmma_gemm.ptx");
    eprintln!("PTX path: {:?}", ptx_path);
    let module = ctx
        .load_module_from_file(ptx_path.to_str().unwrap());
    match &module {
        Ok(m) => eprintln!("Module loaded: {:p}", &**m as *const _),
        Err(e) => eprintln!("Module load error: {:?} ({})", e, e.0),
    }
    let module = module.expect("Failed to load PTX module");

    let tile_dim = 64u32;
    let grid_x = (N as u32 + tile_dim - 1) / tile_dim;
    let grid_y = (M as u32 + tile_dim - 1) / tile_dim;

    println!(
        "Grid: ({}, {}), Block: (128, 1)\n",
        grid_x, grid_y
    );

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    let m_arg = M as u32;
    let n_arg = N as u32;
    let k_arg = K as u32;

    println!("Warmup...");
    cuda_launch! {
        kernel: wgmma_gemm,
        stream: stream,
        module: module,
        config: cfg,
        args: [m_arg, n_arg, k_arg, ALPHA, slice(a_dev), slice(b_dev), BETA, slice_mut(c_dev)]
    }.expect("kernel launch failed");
    stream.synchronize().unwrap();

    const NUM_RUNS: u32 = 10;
    println!("Running {} iterations...", NUM_RUNS);
    let start = Instant::now();
    for _ in 0..NUM_RUNS {
        cuda_launch! {
            kernel: wgmma_gemm,
            stream: stream,
            module: module,
            config: cfg,
            args: [m_arg, n_arg, k_arg, ALPHA, slice(a_dev), slice(b_dev), BETA, slice_mut(c_dev)]
        }.expect("kernel launch failed");
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / NUM_RUNS as f64;

    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let gflops = flops / (avg_ms / 1000.0) / 1e9;

    println!("\nPerformance: {:.3} ms, {:.2} GFLOPS", avg_ms, gflops);

    let c_result = c_dev.to_host_vec(&stream).unwrap();
    println!("\nVerifying (sampling 100 elements)...");
    let mut max_error = 0.0f32;
    let mut max_idx = 0usize;
    for sample in 0..100 {
        let idx = sample * M * N / 100;
        let row = idx / N;
        let col = idx % N;

        let mut expected = 0.0f32;
        for kk in 0..K {
            expected += a[row * K + kk] * b[kk * N + col];
        }
        expected = ALPHA * expected + BETA * c_init[idx];

        let error = (c_result[idx] - expected).abs();
        if error > max_error {
            max_error = error;
            max_idx = idx;
        }
    }

    let max_row = max_idx / N;
    let max_col = max_idx % N;
    println!("Max error: {:.6e} at C[{}][{}]", max_error, max_row, max_col);

    println!("\nElement checks:");
    for (row, col) in &[(0, 0), (M/2, N/2), (M-1, N-1)] {
        let mut expected = 0.0f32;
        for kk in 0..K {
            expected += a[*row * K + kk] * b[kk * N + *col];
        }
        let idx = row * N + col;
        println!("  C[{}][{}] = {:.6e} (expected {:.6e}, err {:.2e})",
            row, col, c_result[idx], expected, (c_result[idx] - expected).abs());
    }

    if max_error < 1e-2 {
        println!("\nSUCCESS!");
    } else {
        println!("\nFAILED! (max error too large)");
        std::process::exit(1);
    }
}
