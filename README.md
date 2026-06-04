# llm-workspace

LLM inference stack: runner, plug-in protocol, safetensors, GGUF parser.

## Workspace Members

| Crate | Description | Status |
|-------|-------------|--------|
| `gguf` | GGUF model weight file parser: header, tensor metadata, KV config, quantization type detection | Working |
| `gguf-cli` | CLI tool to inspect GGUF model files | Working |
| `safetensors` | SQLite-backed weight storage, safetensors parser, GGUF-to-safetensors conversion | Working |
| `llm-plug-in` | Weight manifest generation, inference request/response protocol, prompt templates | Working |
| `llm-runner` | Inference engine: tensor computation, model loading, tokenizer, device backend | Partial |

## Requirements

- Rust nightly (pinned in `rust-toolchain.toml`) — required for cuda-oxide integration
- CUDA Toolkit (for GPU path; CPU fallback available)

## Building

```bash
cargo check --workspace
cargo build -p crabjar-llm-runner
cargo test --workspace
```

## GGUF CLI

```bash
cargo run -p crabjar-gguf-cli -- inspect <file.gguf>
cargo run -p crabjar-gguf-cli -- list <file.gguf>
cargo run -p crabjar-gguf-cli -- tensor <file.gguf> -t
cargo run -p crabjar-gguf-cli -- tensor <file.gguf> -e <name>
```

## CUDA-Oxide Dependencies

Workspace includes path dependencies to cuda-oxide crates (not published):

- `cuda-core` — safe RAII wrappers around CUDA driver API
- `cuda-device` — device-side intrinsics and types for CUDA kernels
- `cuda-host` — host-side utilities for CUDA kernel development
- `cuda-macros` — procedural macros for CUDA kernels
- `cuda-bindings` — raw FFI bindings to CUDA driver API
- `cuda-async` — async execution layer for CUDA device operations
- `libnvvm-sys` — runtime bindings to NVIDIA libNVVM
- `nvjitlink-sys` — runtime bindings to NVIDIA nvJitLink
- `reserved-oxide-symbols` — internal symbol-name contract

`rustc-codegen-cuda` is excluded from the workspace — it requires `#![feature(rustc_private)]` and is built as a dylib rustc codegen backend.

## Current State

GGUF parsing, safetensors storage, and weight loading are functional. The transformer model architecture (`LlamaModel`) is implemented: Q/K/V projections, multi-head attention (CPU), FFN with SwiGLU, RMSNorm, RoPE, LM head, and end-to-end `forward()` all wired. The GPU kernel path (tcgen05/WGMMA) is stubbed. Remaining: tokenizer wiring, sampling integration, and connecting the GPU-focused `Model::run()` loop to `LlamaModel` weights. See `ROADMAP.md` for the implementation plan.

## License

MIT OR Apache-2.0 (except `gguf/` which is AGPL-3.0-or-later)
