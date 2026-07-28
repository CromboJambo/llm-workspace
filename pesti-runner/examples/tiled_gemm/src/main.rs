//! Tiled GEMM for sm_120 (consumer Blackwell — RTX 5060 Ti / 5090)
//!
//! Phase 1: shared memory tiling with f16 inputs.
//! This is the foundation kernel — all later phases (WGMMA, tcgen05) build on this structure.
//!
//! Build and run with:
//!   cargo oxide run tiled_gemm

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread, sync_threads, SharedArray};
use cuda_host::cuda_launch;
use half::f16;
use std::time::Instant;

const TILE: usize = 16;

#[kernel]
pub fn sgemm_tiled(
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a: &[f16],
    b: &[f16],
    beta: f32,
    mut c: DisjointSlice<f32>,
) {
    static mut TILE_A: SharedArray<f16, { TILE * TILE }> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f16, { TILE * TILE }> = SharedArray::UNINIT;

    let row = thread::index_2d_row();
    let col = thread::index_2d_col();
    let tid = thread::threadIdx_x() as usize;
    let bx = thread::blockIdx_x();
    let by = thread::blockIdx_y();

    let c_row = by as u32 * TILE as u32 + row as u32;
    let c_col = bx as u32 * TILE as u32 + col as u32;

    let ty = tid / TILE;
    let tx = tid % TILE;

    let mut sum: f32 = 0.0;

    let nk = (k as usize + TILE - 1) / TILE;
    for t in 0..nk {
        let a_col_global = t * TILE + tx;
        let a_val_raw = if c_row < m && a_col_global < k as usize {
            a[(c_row as usize) * (k as usize) + a_col_global]
        } else {
            f16::ZERO
        };
        unsafe { TILE_A[tid] = a_val_raw };

        let b_row_global = t * TILE + ty;
        let b_val_raw = if b_row_global < k as usize && c_col < n {
            b[b_row_global * (n as usize) + (c_col as usize)]
        } else {
            f16::ZERO
        };
        unsafe { TILE_B[tid] = b_val_raw };

        sync_threads();

        let mut inner_sum = 0.0f32;
        for i in 0..TILE {
            let a_val = unsafe { TILE_A[ty * TILE + i] };
            let b_val = unsafe { TILE_B[i * TILE + tx] };
            inner_sum += a_val.to_f32() * b_val.to_f32();
        }
        sum += inner_sum;

        sync_threads();
    }

    if c_row < m && c_col < n {
        if let Some(c_idx) = thread::index_2d(n as usize) {
            if let Some(c_elem) = c.get_mut(c_idx) {
                *c_elem = alpha * sum + beta * (*c_elem);
            }
        }
    }
}

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;
const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

fn main() {
    println!("=== Tiled GEMM (sm_120 / consumer Blackwell) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    println!("Matrix dimensions: {}x{} * {}x{} = {}x{}", M, K, K, N, M, N);
    println!("Tile size: {}x{}", TILE, TILE);
    println!("alpha = {}, beta = {}\n", ALPHA, BETA);

    let mut a = vec![f16::ZERO; M * K];
    let mut b = vec![f16::ZERO; K * N];
    let c_init = vec![0.0f32; M * N];

    for i in 0..M {
        for j in 0..K {
            a[i * K + j] = f16::from_f32(((i + j) % 127) as f32 / 127.0);
        }
    }
    for i in 0..K {
        for j in 0..N {
            b[i * N + j] = f16::from_f32(((i * j) % 127) as f32 / 127.0);
        }
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut c_dev = DeviceBuffer::from_host(&stream, &c_init).unwrap();

    let module = ctx.load_module_from_file("tiled_gemm.ptx").expect("Failed to load PTX");

    let block_size = TILE as u32;
    let grid_x = (N as u32 + block_size - 1) / block_size;
    let grid_y = (M as u32 + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_size, block_size, 1),
        shared_mem_bytes: 0,
    };

    let m_arg = M as u32;
    let n_arg = N as u32;
    let k_arg = K as u32;

    println!("Grid: ({}, {}), Block: ({}, {})", grid_x, grid_y, block_size, block_size);

    // Warmup
    println!("\nWarmup...");
    cuda_launch! {
        kernel: sgemm_tiled,
        stream: stream,
        module: module,
        config: cfg,
        args: [m_arg, n_arg, k_arg, ALPHA, slice(a_dev), slice(b_dev), BETA, slice_mut(c_dev)]
    }.expect("Kernel launch failed");
    stream.synchronize().unwrap();

    // Benchmark
    const NUM_RUNS: u32 = 10;
    println!("Running {} iterations...", NUM_RUNS);
    let start = Instant::now();
    for _ in 0..NUM_RUNS {
        cuda_launch! {
            kernel: sgemm_tiled,
            stream: stream,
            module: module,
            config: cfg,
            args: [m_arg, n_arg, k_arg, ALPHA, slice(a_dev), slice(b_dev), BETA, slice_mut(c_dev)]
        }.expect("Kernel launch failed");
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / NUM_RUNS as f64;
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let gflops = flops / (elapsed.as_secs_f64()) / 1e9;

    println!("\nPerformance: {:.3} ms, {:.2} GFLOPS", avg_ms, gflops);

    // Verify against naive host computation
    let c_result = c_dev.to_host_vec(&stream).unwrap();
    println!("\nVerifying (sampling 100 elements against naive host GEMM)...");
    let mut max_error = 0.0f32;
    let mut worst_idx = 0usize;
    for sample in 0..100 {
        let idx = sample * M * N / 100;
        let row = idx / N;
        let col = idx % N;

        let mut expected = 0.0f32;
        for kk in 0..K {
            expected += a[row * K + kk].to_f32() * b[kk * N + col].to_f32();
        }
        expected = ALPHA * expected + BETA * c_init[idx];

        let error = (c_result[idx] - expected).abs();
        if error > max_error {
            max_error = error;
            worst_idx = idx;
        }
    }

    let worst_row = worst_idx / N;
    let worst_col = worst_idx % N;
    let mut expected_worst = 0.0f32;
    for kk in 0..K {
        expected_worst += a[worst_row * K + kk].to_f32() * b[kk * N + worst_col].to_f32();
    }

    println!("Max error: {:.6e} at C[{}][{}]", max_error, worst_row, worst_col);
    println!("  Expected: {:.6e}, Got: {:.6e}", expected_worst, c_result[worst_idx]);

    // Also verify a few specific elements
    println!("\nSpecific element checks:");
    let test_positions = [(0, 0), (1, 0), (100, 200), (512, 512), (1023, 1023)];
    for (r, c) in test_positions {
        let idx = r * N + c;
        let mut expected = 0.0f32;
        for kk in 0..K {
            expected += a[r * K + kk].to_f32() * b[kk * N + c].to_f32();
        }
        let error = (c_result[idx] - expected).abs();
        println!("  C[{}][{}] = {:.6e} (expected {:.6e}, err {:.2e})",
            r, c, c_result[idx], expected, error);
    }

    if max_error < 1e-3 {
        println!("\nSUCCESS!");
    } else {
        println!("\nFAILED! (max error too large)");
        std::process::exit(1);
    }
}
