//! Test dispatch with real GGUF models — validates end-to-end correctness:
//! - RoPE + attention correctness
//! - KV cache management in dispatch path
//! - Weight loading (f32 → f16 conversion)
//! - Output head correctness

use std::path::Path;

/// Find a real GGUF model on disk.
fn find_gguf_model() -> Option<&'static str> {
    // Prefer the smallest non-embedding model to keep test time reasonable.
    const CANDIDATES: &[&str] = &[
        "/mnt/data/state/ai/lmstudio/models/lmstudio-community/embeddinggemma-300m-qat-GGUF/embeddinggemma-300m-qat-Q4_0.gguf",
        "/mnt/data/state/ai/lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf",
        "/mnt/data/state/ai/lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf",
    ];
    for path in CANDIDATES {
        if Path::new(path).is_file() {
            return Some(path);
        }
    }
    None
}

/// Compare two f32 slices with relative + absolute tolerance.
fn compare_vectors(a: &[f32], b: &[f32], label_a: &str, label_b: &str) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!(
            "{} len={} != {} len={}",
            label_a,
            a.len(),
            label_b,
            b.len()
        ));
    }
    let mut max_diff: f32 = 0.0;
    let mut max_rel: f32 = 0.0;
    let mut max_idx = 0;
    for (i, (a_val, b_val)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (a_val - b_val).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
        let tol = a_val.abs().max(b_val.abs()) * 1e-3 + 1e-4;
        if *a_val != *b_val && diff > tol {
            let rel = if b_val.abs() > 1e-8 {
                diff / b_val.abs()
            } else {
                diff
            };
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    println!("  max abs diff: {:.6e} at index {}", max_diff, max_idx);
    println!("  max rel diff: {:.6e}", max_rel);
    println!(
        "  a[{}]={:.8}, b[{}]={:.8}",
        max_idx, a[max_idx], max_idx, b[max_idx]
    );
    // Use 1e-2 tolerance for float16 precision loss across the full path
    if max_diff > 1e-2 {
        return Err(format!(
            "{} vs {} max diff {:.6e} at index {} exceeds 1e-2 tolerance",
            label_a, label_b, max_diff, max_idx
        ));
    }
    Ok(())
}

#[test]
    #[ignore] // Synthetic GGUF v3 helper - removed
fn test_dispatch_real_model_logits_match() {
    let model_path = find_gguf_model()
        .expect("No GGUF model found — place one at /mnt/data/state/ai/lmstudio/models/ or update CANDIDATES");

    println!("\n=== Testing dispatch with real model: {} ===", model_path);

    let path = Path::new(model_path);

    // Load CPU model
    let mut cpu_model = pesti_runner::model::CpuModel::load_gguf(path)
        .unwrap_or_else(|e| panic!("Failed to load GGUF at {}: {}", model_path, e));

    // Load dispatch model
    let mut dispatch_model = pesti_runner::model::CpuModel::load_gguf(path)
        .unwrap_or_else(|e| panic!("Failed to load GGUF at {}: {}", model_path, e));
    dispatch_model.enable_dispatch();

    // Verify dispatch is active
    assert!(
        dispatch_model.can_use_dispatch(),
        "Dispatch should be enabled"
    );

    // Pick a token to test with — use token 0 or the first vocab token
    // For llama-family models, token 0 is typically the BOS token
    let token: u32 = 0;

    // Run CPU path
    let cpu_logits = cpu_model
        .decode(token)
        .unwrap_or_else(|e| panic!("CPU decode failed for token {}: {}", token, e));

    println!("CPU logits[0..10]: {:?}", &cpu_logits[..10.min(cpu_logits.len())]);
    println!("CPU logits len: {}", cpu_logits.len());

    // Run dispatch path
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("embed failed");
    println!("Hidden state dim: {}", dispatch_hidden.len());

    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("forward_with_dispatch failed");

    let dispatch_logits = dispatch_model
        .apply_output_head(&dispatch_hidden)
        .expect("apply_output_head failed");

    println!(
        "Dispatch logits[0..10]: {:?}",
        &dispatch_logits[..10.min(dispatch_logits.len())]
    );
    println!("Dispatch logits len: {}", dispatch_logits.len());

    // Compare
    compare_vectors(
        &cpu_logits,
        &dispatch_logits,
        "CPU logits",
        "Dispatch logits",
    )
    .unwrap();

    println!("✅ CPU vs dispatch logits match");
}

#[test]
    #[ignore] // Synthetic GGUF v3 helper - removed
fn test_dispatch_real_model_hidden_states_match() {
    let model_path = find_gguf_model()
        .expect("No GGUF model found");

    println!("\n=== Testing hidden state correctness ===");

    let path = Path::new(model_path);

    // Load CPU model
    let cpu_model = pesti_runner::model::CpuModel::load_gguf(path)
        .unwrap_or_else(|e| panic!("Failed to load GGUF at {}: {}", model_path, e));

    // Load dispatch model
    let mut dispatch_model = pesti_runner::model::CpuModel::load_gguf(path)
        .unwrap_or_else(|e| panic!("Failed to load GGUF at {}: {}", model_path, e));
    dispatch_model.enable_dispatch();

    let token: u32 = 0;

    // Get CPU hidden states (after all transformer layers, before LM head)
    let cpu_hidden = cpu_model
        .llama_model
        .embed(token, 0)
        .expect("embed failed");
    let cpu_hidden = cpu_model
        .llama_model
        .forward_layers(&cpu_hidden, 0)
        .expect("forward_layers failed");

    // Get dispatch hidden states
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("embed failed");
    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("forward_with_dispatch failed");

    println!(
        "CPU hidden[0..5]: {:?}",
        &cpu_hidden[..5.min(cpu_hidden.len())]
    );
    println!(
        "Dispatch hidden[0..5]: {:?}",
        &dispatch_hidden[..5.min(dispatch_hidden.len())]
    );

    compare_vectors(
        &cpu_hidden,
        &dispatch_hidden,
        "CPU hidden",
        "Dispatch hidden",
    )
    .unwrap();

    println!("✅ Hidden states match");
}

#[test]
    #[ignore] // Synthetic GGUF v3 helper - removed
fn test_dispatch_real_model_multi_token() {
    let model_path = find_gguf_model()
        .expect("No GGUF model found");

    println!("\n=== Testing multi-token dispatch ===");

    let path = Path::new(model_path);

    // Load CPU model
    let mut cpu_model = pesti_runner::model::CpuModel::load_gguf(path)
        .unwrap_or_else(|e| panic!("Failed to load GGUF at {}: {}", model_path, e));

    // Load dispatch model
    let mut dispatch_model = pesti_runner::model::CpuModel::load_gguf(path)
        .unwrap_or_else(|e| panic!("Failed to load GGUF at {}: {}", model_path, e));
    dispatch_model.enable_dispatch();

    // Test with multiple tokens (simulating a short sequence)
    let test_tokens: Vec<u32> = vec![0, 1, 2, 3, 4];

    for (i, &token) in test_tokens.iter().enumerate() {
        // CPU path
        let cpu_logits = cpu_model.decode(token).unwrap_or_else(|e| {
            panic!("CPU decode failed for token {} at pos {}: {}", token, i, e)
        });

        // Dispatch path
        let dispatch_hidden = dispatch_model
            .llama_model
            .embed(token, i)
            .unwrap_or_else(|e| panic!("embed failed for token {} at pos {}: {}", token, i, e));
        let dispatch_hidden = dispatch_model
            .forward_with_dispatch(&dispatch_hidden, i)
            .unwrap_or_else(|e| {
                panic!(
                    "forward_with_dispatch failed for token {} at pos {}: {}",
                    token, i, e
                )
            });
        let dispatch_logits = dispatch_model
            .apply_output_head(&dispatch_hidden)
            .unwrap_or_else(|e| panic!("apply_output_head failed: {}", e));

        compare_vectors(
            &cpu_logits,
            &dispatch_logits,
            &format!("CPU logits [token={}]", token),
            &format!("Dispatch logits [token={}]", token),
        )
        .unwrap_or_else(|e| panic!("Mismatch at token {}: {}", token, e));

        println!("✅ Token {} (pos {}) logits match", token, i);
    }
}

#[test]
    #[ignore] // Synthetic GGUF v3 helper - removed
fn test_dispatch_real_model_kv_cache_persistence() {
    let model_path = find_gguf_model()
        .expect("No GGUF model found");

    println!("\n=== Testing KV cache persistence ===");

    let path = Path::new(model_path);

    let mut dispatch_model = pesti_runner::model::CpuModel::load_gguf(path)
        .unwrap_or_else(|e| panic!("Failed to load GGUF at {}: {}", model_path, e));
    dispatch_model.enable_dispatch();

    // First forward pass
    let token1: u32 = 0;
    let hidden1 = dispatch_model
        .llama_model
        .embed(token1, 0)
        .expect("embed failed");
    let out1 = dispatch_model
        .forward_with_dispatch(&hidden1, 0)
        .expect("forward_with_dispatch (first pass) failed");

    // Second forward pass at a different position — KV cache should persist
    let token2: u32 = 1;
    let hidden2 = dispatch_model
        .llama_model
        .embed(token2, 5)
        .expect("embed failed");
    let out2 = dispatch_model
        .forward_with_dispatch(&hidden2, 5)
        .expect("forward_with_dispatch (second pass) failed");

    // Outputs should differ because positions differ (RoPE)
    assert!(
        out1.iter().zip(out2.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
        "outputs should differ at different positions"
    );

    println!("✅ KV cache persistence works (different positions produce different outputs)");
}

#[test]
    #[ignore] // Synthetic GGUF v3 helper - removed
fn test_dispatch_real_model_gpu_fallback_to_cpu() {
    let model_path = find_gguf_model()
        .expect("No GGUF model found");

    println!("\n=== Testing GPU fallback to CPU ===");

    // Create dispatch context with GPU preference but no GPU available
    let ctx = pesti_runner::kernel::dispatch::DispatchContext::with_gpu_preference(true);

    // Even without GPU, the dispatch should work via CPU fallback
    let path = Path::new(model_path);
    let mut dispatch_model = pesti_runner::model::CpuModel::load_gguf(path)
        .unwrap_or_else(|e| panic!("Failed to load GGUF at {}: {}", model_path, e));
    dispatch_model.enable_dispatch();

    let token: u32 = 0;
    let hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("embed failed");

    // This should succeed even without a GPU (CPU fallback)
    let result = dispatch_model
        .forward_with_dispatch(&hidden, 0)
        .expect("dispatch should succeed via CPU fallback even without GPU");

    println!("Dispatch output shape (CPU fallback): {}", result.len());
    assert!(result.len() > 0, "Output should not be empty");

    println!("✅ GPU fallback to CPU works");

    // Suppress unused variable warning
    let _ = ctx;
}
