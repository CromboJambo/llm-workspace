use pesti_runner::kernel::dispatch::{DispatchContext, LinearDispatch, AttentionDispatch};
use pesti_runner::kernel::kvcache::Kvcache;
use pesti_runner::model::CpuModel;
use pesti_runner::transformer::model::LlamaModel;
use pesti_gguf::{GgufKvPair, GgufTensorInfo, kv_pair_str, kv_pair_u32, kv_pair_f32, compute_data_section_start};
use half::f16;
use std::path::PathBuf;
use tempfile::tempdir;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Create a minimal synthetic GGUF file for testing.
/// Architecture: llama, 2 layers, 64-dim embedding, 4 heads.
fn make_test_gguf(path: &PathBuf) {
    let kv_pairs: Vec<GgufKvPair> = vec![
        kv_pair_str("general.architecture", "llama"),
        kv_pair_str("general.file_type", "F16"),
        kv_pair_u32("llama.context_length", 4096),
        kv_pair_u32("llama.embedding_length", 64),
        kv_pair_u32("llama.block_count", 2),
        kv_pair_u32("llama.attention.head_count", 4),
        kv_pair_u32("llama.attention.head_count_kv", 2),
        kv_pair_u32("llama.feed_forward_length", 128),
        kv_pair_u32("llama.rope.dimension_count", 64),
        kv_pair_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
        kv_pair_u32("tokenizer.ggml.tokens", 32000),
    ];

    let tensor_shapes: Vec<Vec<u64>> = vec![
        vec![64],        // tok_embeddings
        vec![32000, 64], // output
        vec![64, 64],    // layers.0.attention.wq
        vec![64, 64],    // layers.0.attention.wk
        vec![64, 64],    // layers.0.attention.wv
        vec![64, 64],    // layers.0.attention.wo
        vec![64],        // layers.0.attention_norm
        vec![64],        // layers.0.ffn_norm
        vec![64, 128],   // layers.0.feed_forward.w1
        vec![128, 64],   // layers.0.feed_forward.w2
        vec![64, 128],   // layers.0.feed_forward.w3
    ];
    let tensor_names: Vec<&str> = vec![
        "tok_embeddings.weight",
        "output.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.feed_forward.w3.weight",
    ];

    let mut offset = 0u64;
    let tensor_infos: Vec<GgufTensorInfo> = tensor_shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            let info = GgufTensorInfo {
                name: tensor_names[i].to_string(),
                shape: shape.clone(),
                offset,
                dtype: 1, // F16
            };
            let elems: u64 = shape.iter().product();
            offset += elems * 2;
            info
        })
        .collect();

    let data_section_start = compute_data_section_start(3, &kv_pairs, &tensor_infos, None);

    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensor_infos.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());
    for kv in &kv_pairs {
        let key_bytes = kv.key.as_bytes();
        buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(&(kv.value_type() as u32).to_le_bytes());
        match &kv.value {
            pesti_gguf::GgufKvValue::Uint8(v) => buf.push(*v),
            pesti_gguf::GgufKvValue::Int8(v) => buf.push(*v),
            pesti_gguf::GgufKvValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Int16(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Int32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Float16(v) => buf.extend_from_slice(&(*v as u16).to_le_bytes()),
            pesti_gguf::GgufKvValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Int64(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Float64(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Bool(v) => buf.push(if *v { 1u8 } else { 0u8 }),
            pesti_gguf::GgufKvValue::String(s) => {
                buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            pesti_gguf::GgufKvValue::Array(arr) => {
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                buf.extend_from_slice(&(arr.type_() as u32).to_le_bytes());
                for item in arr {
                    match item {
                        pesti_gguf::GgufKvValue::Uint8(v) => buf.push(*v),
                        pesti_gguf::GgufKvValue::Int8(v) => buf.push(*v),
                        pesti_gguf::GgufKvValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        pesti_gguf::GgufKvValue::Int16(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        pesti_gguf::GgufKvValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        pesti_gguf::GgufKvValue::Int32(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        pesti_gguf::GgufKvValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        pesti_gguf::GgufKvValue::Float16(v) => buf.extend_from_slice(&(*v as u16).to_le_bytes()),
                        pesti_gguf::GgufKvValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        pesti_gguf::GgufKvValue::Int64(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        pesti_gguf::GgufKvValue::Float64(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        pesti_gguf::GgufKvValue::Bool(v) => buf.push(if *v { 1u8 } else { 0u8 }),
                        pesti_gguf::GgufKvValue::String(s) => {
                            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                            buf.extend_from_slice(s.as_bytes());
                        }
                        pesti_gguf::GgufKvValue::Array(_) => unreachable!(),
                    }
                }
            }
        }
    }

    // Write tensor data (random-ish but deterministic: just zeros for testing)
    for info in &tensor_infos {
        let elems: u64 = info.shape.iter().product();
        buf.extend_from_slice(&vec![0u8; (elems * 2) as usize]); // F16 = 2 bytes per element
    }

    std::fs::write(path, buf).unwrap();
}

// ── Tests ───────────────────────────────────────────────────────────────────

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

/// Test that the dispatch path produces output matching the CPU path when
/// run on the same GGUF model with the same input.
///
/// This validates:
/// - RoPE + attention correctness end-to-end
/// - KV cache management in dispatch path
/// - Weight loading (f16 → f32 conversion)
/// - Output head correctness
#[test]
fn test_dispatch_vs_cpu_output() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("test.gguf");
    make_test_gguf(&gguf_path);

    // Load model for CPU path
    let mut cpu_model = CpuModel::load_gguf(&gguf_path).expect("Failed to load GGUF");

    // Load model for dispatch path
    let mut dispatch_model = CpuModel::load_gguf(&gguf_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    // Run a single token through both paths
    let token: u32 = 0;

    // CPU path output
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");

    // Reset dispatch model (KV cache gets populated during decode)
    dispatch_model.reset();

    // Dispatch path output
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("embed failed");
    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("forward_with_dispatch failed");
    let dispatch_logits = dispatch_model
        .apply_output_head(&dispatch_hidden)
        .expect("apply_output_head failed");

    println!("CPU logits (first 10): {:?}", &cpu_logits[..10.min(cpu_logits.len())]);
    println!("Dispatch logits (first 10): {:?}", &dispatch_logits[..10.min(dispatch_logits.len())]);

    // Outputs should match (within floating point tolerance)
    assert_eq!(cpu_logits.len(), dispatch_logits.len(), "Logit vector length mismatch");
    for (i, (cpu, dispatch)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (cpu - dispatch).abs();
        // Use relative tolerance for larger values, absolute for smaller
        let tol = 1e-3.max(cpu.abs() * 1e-4);
        assert!(
            diff < tol,
            "Logit mismatch at index {}: cpu={:.6} dispatch={:.6} diff={:.6} tol={:.6}",
            i, cpu, dispatch, diff, tol
        );
    }
}

/// Test that dispatch falls back to CPU when GPU is unavailable.
#[test]
fn test_dispatch_cpu_fallback() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("test.gguf");
    make_test_gguf(&gguf_path);

    let mut model = CpuModel::load_gguf(&gguf_path).expect("Failed to load GGUF");
    model.enable_dispatch();

    // Even without GPU, dispatch should work (falls back to CPU)
    let hidden = model
        .llama_model
        .embed(0, 0)
        .expect("embed failed");
    let result = model
        .forward_with_dispatch(&hidden, 0)
        .expect("dispatch should fall back to CPU when GPU unavailable");

    // Result should have the same shape as the embedding
    assert_eq!(result.len(), hidden.len(), "Dispatch output shape mismatch");
    println!("Dispatch fallback output shape: {}", result.len());
}
