use crate::parser::parse_gguf;
use std::path::PathBuf;

#[test]
fn test_parse_conformance_corpus_qwen2_5() {
    let path = PathBuf::from("/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    
    // Should parse without error
    let header = parse_gguf(&path).expect("Failed to parse real GGUF file");

    println!("Version: {}", header.version);
    println!("KV pairs: {}", header.kv_pairs.len());
    println!("Tensors: {}", header.tensors.len());

    // Verify we got at least some data
    assert!(header.kv_pairs.len() > 0, "Should have KV pairs");
    assert!(header.tensors.len() > 0, "Should have tensors");

    // Check a specific key exists
    let has_architecture = header.kv_pairs.iter().any(|p| p.key == "general.architecture");
    assert!(has_architecture, "Should have general.architecture KV pair");

    println!("SUCCESS: Real GGUF file parsed correctly!");
}
