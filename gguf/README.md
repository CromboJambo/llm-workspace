# crabjar-gguf

GGUF model weight file parser for Rust.

Parses GGUF v1/v2/v3 headers, tensor metadata, KV config, and quantization type detection.

## Usage

```toml
[dependencies]
crabjar-gguf = "0.1"
```

```rust
use crabjar_gguf::parse_gguf;

let header = parse_gguf(&path).unwrap();
println!("architecture: {}", header.architecture().unwrap());
println!("tensors: {}", header.tensors.len());
```

## Features

- GGUF v1, v2, v3 parsing
- All quantization types (Q4_0 through Q8_K, K-family variants)
- Tensor stored size computation
- Key-value config extraction with architecture-specific fallback keys
- Serialization (serde) support for all types
- Zero-copy tensor byte extraction

## License

AGPL-3.0-or-later
