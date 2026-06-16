use pesti_runner::kernel::dispatch::{DispatchContext, LinearDispatch, AttentionDispatch};
use pesti_runner::kernel::kvcache::Kvcache;
use half::f16;

#[test]
fn test_dispatch_context_gpu_detection() {
    let ctx = DispatchContext::new();
    // Verify that we are aware of the GPU state
    println!("Prefer GPU: {}", ctx.prefer_gpu());
    println!("GPU Available: {}", ctx.gpu_available());
    println!("Device Info: {}", ctx.device_info());
}

#[test]
fn test_linear_dispatch_accuracy() {
    // Mock data: 1x2 input, 2x2 weights
    let x = vec![1.0f32, 2.0f32];
    let weights_f16 = vec![
        f16::from_f32(1.0),
        f16::from_f32(0.5),
        f16::from_f32(0.5),
        f16::from_f32(1.0),
    ];
    let weights_f32 = vec![1.0f32, 0.5f32, 0.5f32, 1.0f32];
    let bias = Some(vec![0.1f32, 0.1f32]);

    let linear = LinearDispatch::new(weights_f16, weights_f32, bias, 2, 2);
    let ctx = DispatchContext::new();

    // Run via dispatch (GPU or CPU)
    let result = linear.forward(&ctx, &x, 1).expect("Linear dispatch failed");
    println!("linear result: {:?}", result);

    // Manual calculation (row-major weights [o*in+i]):
    // row 0: x[0]*w[0] + x[1]*w[1] + bias[0] = 1.0*1.0 + 2.0*0.5 + 0.1 = 2.1
    // row 1: x[0]*w[2] + x[1]*w[3] + bias[1] = 1.0*0.5 + 2.0*1.0 + 0.1 = 2.6
    assert!((result[0] - 2.1).abs() < 1e-4);
    assert!((result[1] - 2.6).abs() < 1e-4);
}

/// Attention dispatch test requires a populated KV cache, but Kvcache has no
/// public API to write values (buffer is DeviceBuffer<f16> behind a trait).
/// Skipping until we add a test-only populate method or use a real model path.
#[test]
#[ignore = "requires Kvcache write API for test setup"]
fn test_attention_dispatch_mock() {
    // Minimal dimensions to test the flow
    let num_heads = 1;
    let head_dim = 2;
    let num_kv_heads = 1;
    let max_seq = 2;

    let wq = LinearDispatch::new(vec![f16::from_f32(1.0); 4], vec![1.0f32; 4], None, 2, 2);
    let wk = LinearDispatch::new(vec![f16::from_f32(1.0); 4], vec![1.0f32; 4], None, 2, 2);
    let wv = LinearDispatch::new(vec![f16::from_f32(1.0); 4], vec![1.0f32; 4], None, 2, 2);
    let wo = LinearDispatch::new(vec![f16::from_f32(1.0); 4], vec![1.0f32; 4], None, 2, 2);

    let attention = AttentionDispatch {
        wq,
        wk,
        wv,
        wo,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim: 2,
    };

    let ctx = DispatchContext::new();
    let x = vec![1.0f32, 1.0f32]; // batch_size=1, seq_len=1

    // Create separate key and value KV caches (current API requires both)
    let key_cache = Kvcache::new(num_heads, head_dim, max_seq, false);
    let value_cache = Kvcache::new(num_heads, head_dim, max_seq, false);

    let result = attention
        .forward(&ctx, &x, 1, 1, 0, &key_cache, &value_cache)
        .expect("Attention dispatch failed");
    println!("attention result: {:?}", result);

    // With all weights = 1.0, input = 1.0, and RoPE identity (at pos 0):
    // Q, K, V will all be [1.0, 1.0]
    // Attention score = (1.0*1.0 + 1.0*1.0) / sqrt(2) = 2 / 1.414 ≈ 1.414
    // Softmax(1.414) = 1.0
    // Output = 1.0 * V = [1.0, 1.0]
    // Final projection wo = [1.0, 1.0]
    // Result = [1.0, 1.0]
    assert!((result[0] - 1.0).abs() < 1e-2);
    assert!((result[1] - 1.0).abs() < 1e-2);
}
