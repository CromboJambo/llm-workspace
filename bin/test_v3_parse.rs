use std::io::{Read, Seek};

fn main() {
    let mut file = std::fs::File::open("/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf").unwrap();
    
    // Skip header: magic(4) + version(4) + tensor_count(8) + kv_count(8) = 24 bytes
    let mut header_buf = [0u8; 24];
    file.read_exact(&mut header_buf).unwrap();
    
    println!("Header (first 32 bytes):");
    for i in 0..32 {
        file.seek(std::io::SeekFrom::Current(1)).unwrap();
    }
    file.seek(std::io::SeekFrom::Current(-32)).unwrap(); // go back
    
    let mut buf = [0u8; 64];
    file.read_exact(&mut buf).unwrap();
    
    println!("\nBytes after header:");
    for i in 0..buf.len() {
        let b = buf[i];
        let char_display = if (b as char).is_ascii_alphabetic() {
            format!("'{}'", b as char)
        } else {
            format!("{}", b)
        };
        println!("{:3}: 0x{:02x} ({:3}) {}", i, b, b, char_display);
    }
    
    // Find where value_type appears (byte <= 15 or < 32)
    println!("\n\nValue_type candidates (byte <= 15):");
    for i in 0..buf.len() {
        let b = buf[i];
        if b <= 15 {
            println!("  offset {}: value_type candidate!", i);
        }
    }
}
