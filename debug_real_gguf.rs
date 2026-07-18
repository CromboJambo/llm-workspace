// Simple test to parse the real conformance corpus model
fn main() {
    let path = std::path::Path::new("/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    
    match testi_gguf::parser::parse_gguf(path) {
        Ok(header) => {
            println!("✅ Successfully parsed real GGUF file!");
            println!("Version: {}", header.version);
            println!("KV pairs: {}", header.kv_pairs.len());
            println!("Tensors: {}", header.tensors.len());
            
            if let Some(arch) = header.architecture() {
                println!("Architecture: {}", arch);
            }
            
            if let Some(ctx_len) = header.context_length() {
                println!("Context length: {}", ctx_len);
            }
            
            if let Some(embed_dim) = header.embedding_length() {
                println!("Embedding dim: {}", embed_dim);
            }
        }
        Err(e) => {
            eprintln!("❌ Parse error: {:?}", e);
        }
    }
}
