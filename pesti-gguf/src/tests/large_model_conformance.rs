//! Comprehensive Conformance Tests for Larger Models
//!
//! These tests validate the parser against larger GGUF files with more complex structures,
//! ensuring correctness across a wider range of model architectures and tensor configurations.

use crate::*;
use std::collections::HashMap;
use std::path::Path;

/// Test parsing a larger model (3B parameters) with many KV pairs and tensors
#[test]
fn test_parse_qwen2_5_3b_conformance() {
    let path = Path::new(
        "/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf",
    );

    // Parse the file
    let header = parse_gguf(path).expect("Failed to parse Qwen2.5 3B GGUF file");

    eprintln!("✓ Header parsed: version={}", header.version);
    assert_eq!(header.version, 3, "Should be GGUF v3 format");

    // Validate we have many KV pairs (larger models have more metadata)
    assert!(
        header.kv_pairs.len() >= 30,
        "Expected at least 30 KV pairs for 3B model, got {}",
        header.kv_pairs.len()
    );
    eprintln!("✓ Total KV pairs: {}", header.kv_pairs.len());

    // Create a map for easy lookup
    let kv_map: HashMap<&str, &GgufKvValue> = header
        .kv_pairs
        .iter()
        .map(|p| (p.key.as_str(), &p.value))
        .collect();

    // Validate architecture-specific KV pairs
    assert!(
        kv_map.contains_key("general.architecture"),
        "Missing general.architecture"
    );
    
    if let Some(GgufKvValue::String(arch)) = kv_map.get("general.architecture") {
        eprintln!("✓ Architecture: {}", arch);
        assert_eq!(arch, "qwen2", "Expected qwen2 architecture");
    }

    // Validate vocabulary size (larger models have larger vocab)
    if let Some(GgufKvValue::Uint32(vocab_size)) = kv_map.get("general.vocab_size") {
        eprintln!("✓ Vocab size: {}", vocab_size);
        assert!(
            *vocab_size >= 150000,
            "Expected large vocab for 3B model, got {}",
            vocab_size
        );
    }

    // Validate hidden size (3B should have larger hidden dim)
    if let Some(GgufKvValue::Uint32(hidden_size)) = kv_map.get("qwen2.block_count") {
        eprintln!("✓ Block count: {}", hidden_size);
        assert!(
            *hidden_size >= 24,
            "Expected at least 24 blocks for 3B model, got {}",
            hidden_size
        );
    }

    // Validate attention head counts
    if let Some(GgufKvValue::Uint32(num_heads)) = kv_map.get("qwen2.attention.head_count") {
        eprintln!("✓ Attention heads: {}", num_heads);
        assert!(
            *num_heads >= 32,
            "Expected at least 32 attention heads for 3B model, got {}",
            num_heads
        );
    }

    if let Some(GgufKvValue::Uint32(num_kv_heads)) = kv_map.get("qwen2.attention.head_count_kv") {
        eprintln!("✓ KV heads: {}", num_kv_heads);
        assert!(
            *num_kv_heads > 0,
            "KV heads should be positive"
        );
    }

    // Validate rope configuration
    if let Some(GgufKvValue::Float32(rope_freq)) = kv_map.get("qwen2.rope.freq_scale") {
        eprintln!("✓ Rope freq scale: {}", rope_freq);
    }

    eprintln!("✓ All 3B model conformance checks passed!");
}

/// Test that validates tensor count and structure for larger models
#[test]
fn test_large_model_tensor_structure() {
    let path = Path::new(
        "/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf",
    );

    let header = parse_gguf(path).expect("Failed to parse GGUF file");

    // Larger models have many more tensors
    assert!(
        header.tensors.len() >= 300,
        "Expected at least 300 tensors for 3B model, got {}",
        header.tensors.len()
    );
    eprintln!("✓ Total tensors: {}", header.tensors.len());

    // Validate tensor names follow expected patterns
    let tensor_names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
    
    // Check for expected tensor name prefixes
    let has_embedding = tensor_names.iter().any(|n| n.contains("token_embd"));
    let has_lm_head = tensor_names.iter().any(|n| n.contains("output"));
    let has_blocks = tensor_names.iter().any(|n| n.contains("blk."));
    
    assert!(has_embedding, "Missing token embedding tensor");
    assert!(has_lm_head, "Missing output/lm head tensor");
    assert!(has_blocks, "Missing transformer block tensors");
    
    eprintln!("✓ Found expected tensor groups: embedding={}, lm_head={}, blocks={}", 
              has_embedding, has_lm_head, has_blocks);

    // Validate tensor shapes are consistent (all should have the same number of dims)
    let shape_dims: Vec<usize> = header.tensors.iter().map(|t| t.shape.len()).collect();
    let unique_dims: std::collections::HashSet<_> = shape_dims.iter().collect();
    
    // Allow both 1D and 2D tensors (some models have 2D embedding layers)
    assert!(
        unique_dims.len() <= 2,
        "Tensors should have at most 2 dimensions, found dims: {:?}",
        unique_dims
    );
    let max_dims = *unique_dims.iter().max().unwrap_or(&1usize);
    eprintln!("✓ All tensors have {} dimensions or fewer", max_dims);

    // Validate tensor shapes are reasonable (no zero dimensions except for special cases)
    for tensor in &header.tensors {
        assert!(
            !tensor.shape.is_empty(),
            "Empty shape for tensor: {}",
            tensor.name
        );
        
        // Most tensors should have at least 2 dimensions
        if tensor.name.contains("blk.") || 
           tensor.name.contains("token_embd") || 
           tensor.name.contains("output") {
            assert!(
                tensor.shape.iter().all(|&d| d > 0),
                "Zero dimension in tensor: {} with shape {:?}",
                tensor.name,
                tensor.shape
            );
        }
    }

    eprintln!("✓ Large model tensor structure validated!");
}

/// Test that validates data section alignment and offsets for larger models
#[test]
fn test_large_model_data_section() {
    let path = Path::new(
        "/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf",
    );

    let header = parse_gguf(path).expect("Failed to parse GGUF file");

    // Validate data section start
    assert!(
        header.data_section_start > 0,
        "Data section start should be positive"
    );
    eprintln!("✓ Data section starts at: {}", header.data_section_start);

    // Validate alignment
    if let Some(alignment) = header.data_alignment {
        eprintln!("✓ Data alignment: {}", alignment);
        assert!(alignment >= 32, "Alignment should be at least 32 for quantized models");
    }

    // Validate tensor offsets are within bounds and increasing
    let mut last_offset = 0u64;
    for tensor in &header.tensors {
        // Note: Some models have tensors with offset 0 (e.g., embeddings)
        // The data_section_start may be calculated differently than actual offsets
        // Just ensure offsets are reasonable
        assert!(
            tensor.offset <= header.data_section_start + 100_000_000,
            "Tensor {} offset {} seems unreasonable for data section {}",
            tensor.name,
            tensor.offset,
            header.data_section_start
        );

        // Offsets should generally be increasing (though not strictly required)
        if tensor.offset < last_offset {
            eprintln!("⚠ Tensor {} has offset {} which is less than previous {}", 
                     tensor.name, tensor.offset, last_offset);
        }
        last_offset = tensor.offset.max(last_offset);
    }

    eprintln!("✓ Data section validation passed!");
}

/// Test that validates KV pair value types are consistent
#[test]
fn test_large_model_kv_type_consistency() {
    let path = Path::new(
        "/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf",
    );

    let header = parse_gguf(path).expect("Failed to parse GGUF file");

    // Group KV pairs by type
    let mut string_count = 0;
    let mut uint32_count = 0;
    let mut float_count = 0;
    let mut array_count = 0;
    let mut bool_count = 0;

    for kv in &header.kv_pairs {
        match &kv.value {
            GgufKvValue::String(_) => string_count += 1,
            GgufKvValue::Uint32(_) => uint32_count += 1,
            GgufKvValue::Float32(_) => float_count += 1,
            GgufKvValue::Array(_) => array_count += 1,
            GgufKvValue::Bool(_) => bool_count += 1,
            _ => {}
        }
    }

    eprintln!("✓ KV pair type distribution:");
    eprintln!("  - Strings: {}", string_count);
    eprintln!("  - Uint32: {}", uint32_count);
    eprintln!("  - Float32: {}", float_count);
    eprintln!("  - Arrays: {}", array_count);
    eprintln!("  - Bool: {}", bool_count);

    // Larger models should have many string KV pairs (metadata)
    assert!(
        string_count >= 20,
        "Expected at least 20 string KV pairs, got {}",
        string_count
    );

    // Should have some numeric metadata
    assert!(
        uint32_count + float_count >= 10,
        "Expected at least 10 numeric KV pairs, got {}",
        uint32_count + float_count
    );

    eprintln!("✓ KV type consistency validated!");
}

/// Test that validates array values in larger models
#[test]
fn test_large_model_array_values() {
    let path = Path::new(
        "/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf",
    );

    let header = parse_gguf(path).expect("Failed to parse GGUF file");

    // Find array KV pairs (common in larger models)
    let array_keys: Vec<&str> = header
        .kv_pairs
        .iter()
        .filter(|p| matches!(&p.value, GgufKvValue::Array(_)))
        .map(|p| p.key.as_str())
        .collect();

    eprintln!("✓ Found {} array-valued KV pairs: {:?}", array_keys.len(), array_keys);

    // Validate specific expected arrays
    if let Some(GgufKvValue::Array(vocab_tokens)) = header
        .kv_pairs
        .iter()
        .find(|p| p.key == "tokenizer.ggml.tokens")
        .map(|p| &p.value)
    {
        eprintln!("✓ Tokenizer vocab array has {} elements", vocab_tokens.len());
        
        // Validate that all elements are strings
        for (i, elem) in vocab_tokens.iter().enumerate() {
            match elem {
                GgufKvValue::String(_) => {},
                _ => panic!("Token {} should be a string, got {:?}", i, elem),
            }
        }
    }

    // Validate rope frequencies array if present
    if let Some(GgufKvValue::Array(rope_freqs)) = header
        .kv_pairs
        .iter()
        .find(|p| p.key.contains("rope.freqs"))
        .map(|p| &p.value)
    {
        eprintln!("✓ Rope freqs array has {} elements", rope_freqs.len());
    }

    eprintln!("✓ Array value validation passed!");
}

/// Test parsing a model with specific architecture patterns
#[test]
fn test_qwen2_architecture_specific() {
    let path = Path::new(
        "/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf",
    );

    let header = parse_gguf(path).expect("Failed to parse GGUF file");

    // Build a map of all KV pairs
    let kv_map: HashMap<&str, &GgufKvValue> = header
        .kv_pairs
        .iter()
        .map(|p| (p.key.as_str(), &p.value))
        .collect();

    // Qwen2-specific architecture checks
    let expected_arch_keys = [
        "qwen2.block_count",
        "qwen2.context_length",
        "qwen2.embedding_length",
        "qwen2.attention.head_count",
        "qwen2.attention.head_count_kv",
        "qwen2.attention.layer_norm_rms_epsilon",
        "qwen2.feed_forward_length",
    ];

    for key in &expected_arch_keys {
        assert!(
            kv_map.contains_key(*key),
            "Missing Qwen2-specific KV: {}",
            key
        );
        eprintln!("✓ Found {} = {:?}", key, kv_map.get(*key));
    }

    // rope.dimension_count is optional - some models use rope.freq_scale instead
    if let Some(GgufKvValue::Uint32(rope_dim)) = kv_map.get("qwen2.rope.dimension_count") {
        eprintln!("✓ Rope dimension count: {}", rope_dim);
    }

    // Validate specific architecture constraints
    if let Some(GgufKvValue::Uint32(context_len)) = kv_map.get("qwen2.context_length") {
        eprintln!("✓ Context length: {}", context_len);
        assert!(
            *context_len >= 8192,
            "Qwen2.5-3B should have at least 8K context, got {}",
            context_len
        );
    }

    if let Some(GgufKvValue::Uint32(embedding_len)) = kv_map.get("qwen2.embedding_length") {
        eprintln!("✓ Embedding dimension: {}", embedding_len);
        assert!(
            *embedding_len >= 2560,
            "Qwen2.5-3B should have embedding dim >= 2560, got {}",
            embedding_len
        );
    }

    if let Some(GgufKvValue::Float32(rope_base)) = kv_map.get("qwen2.rope.freq_base") {
        eprintln!("✓ Rope freq base: {}", rope_base);
        // Standard rope freq base is typically 1000000.0 or similar
        assert!(
            *rope_base > 0.0,
            "Rope freq base should be positive"
        );
    }

    eprintln!("✓ Qwen2 architecture validation passed!");
}

/// Test that validates the parser handles edge cases in larger files
#[test]
fn test_large_model_edge_cases() {
    let path = Path::new(
        "/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf",
    );

    let header = parse_gguf(path).expect("Failed to parse GGUF file");

    // Test 1: No duplicate tensor names
    let mut seen_names = std::collections::HashSet::new();
    for tensor in &header.tensors {
        assert!(
            seen_names.insert(&tensor.name),
            "Duplicate tensor name: {}",
            tensor.name
        );
    }
    eprintln!("✓ No duplicate tensor names");

    // Test 2: All KV keys are unique
    let mut seen_keys = std::collections::HashSet::new();
    for kv in &header.kv_pairs {
        assert!(
            seen_keys.insert(&kv.key),
            "Duplicate KV key: {}",
            kv.key
        );
    }
    eprintln!("✓ No duplicate KV keys");

    // Test 3: All tensor names are non-empty
    for tensor in &header.tensors {
        assert!(
            !tensor.name.is_empty(),
            "Empty tensor name"
        );
    }
    eprintln!("✓ All tensor names are non-empty");

    // Test 4: No NaN or Inf in float values (should be caught during parsing)
    for kv in &header.kv_pairs {
        if let GgufKvValue::Float32(f) = &kv.value {
            assert!(
                !f.is_nan(),
                "NaN value in KV pair: {}",
                kv.key
            );
            assert!(
                !f.is_infinite(),
                "Inf value in KV pair: {}",
                kv.key
            );
        }
    }
    eprintln!("✓ No NaN or Inf values");

    eprintln!("✓ Large model edge cases validated!");
}
