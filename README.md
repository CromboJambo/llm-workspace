# llm-workspace

LLM inference stack: runner, plug-in protocol, safetensors, GGUF parser.

## Workspace Members

| Crate | Description | Status |
|-------|-------------|--------|
| `gguf` | GGUF model weight file parser: header, tensor metadata, KV config, quantization type detection | Working |
| `gguf-cli` | CLI tool to inspect GGUF model files | Working |
| `safetensors` | SQLite-backed weight storage, safetensors parser, GGUF-to-safetensors conversion | Working |
| `llm-plug-in` | Weight manifest generation, inference request/response protocol, prompt templates | Working |
| `llm-runner` | Inference engine with two paths: pure-Rust transformer + llama.cpp FFI | Partial |

## Two Inference Paths

### 1. Pure-Rust Transformer Path (`transformer/`)
Full Llama-style model implementation in pure Rust:
- Q/K/V linear projections, multi-head attention (CPU)
- FFN with SwiGLU, RMSNorm, RoPE positional embeddings
- LM head, token sampling (temperature, top-p, top-k)
- Architecture-aware weight loading (llama, mistral, gemma, qwen2, phi3, mixtral, starcoder2)
- `LlamaModel::generate()` — autoregressive generation loop

### 2. llama.cpp FFI Path (`llama/`)
High-level wrapper over llama-cpp-2:
- `LlamaRunner` with builder pattern for context/model config
- Full generation with timing (prompt eval, token eval)
- Chat template application, grammar-constrained decoding
- Session save/load, embeddings
- Configurable sampling (top-k, top-p, min-p, TFS, typical p, repetition penalty)

## GGUF Weight Loading

Full K-family dequantization support:
- Q4_0, Q4_1, Q8_0 — classic quantization
- Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K — K-family (including Q4_K_M, Q5_K_M, Q5_K_S, Q6_K_S, Q8_K_M, Q1_K)
- F32, F16, BF16, I8, I16, I32, I64 — passthrough/conversion

## Requirements

- Rust nightly (pinned in `rust-toolchain.toml`) — required for cuda-oxide integration
- CUDA Toolkit (for GPU path; CPU fallback available)
- llama.cpp (via llama-cpp-2 crate; FFI path)

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

## Device Routing

Hybrid device selector with priority-based routing:
- **Local GPU** → CUDA via candle-core/cudarc
- **Remote LM Studio** → HTTP transport via `RunnerBridge` (health-checked)
- **CPU** → fallback

`DeviceRouter` combines `DeviceSelector` (discovery + priority) with `RunnerBridge` (remote transport) into a single execution pipeline.

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

**Phase 1 (CPU Inference): ✅ Complete**
- Pure-Rust transformer path: full LlamaModel with Q/K/V projections, attention, FFN, RMSNorm, RoPE, sampling, tokenizer — all wired end-to-end
- llama.cpp FFI path: complete `LlamaRunner` with generation, chat, embeddings, grammar, sessions
- GGUF weight loading: all 29+ quantization types implemented (Q1_K through Q8_K_M)

**Phase 1.5 (Hybrid Routing): ✅ Complete**
- `DeviceSelector` with GPU → Remote → CPU priority routing
- `RemoteDevice` discovery via health checks
- `RunnerBridge` for remote LM Studio HTTP transport
- `DeviceRouter` combining discovery + transport
- `ModelManager` with popularity scoring and smart preloading
- `Registry` with in-memory + filesystem model discovery

**Phase 2 (GPU Acceleration): 🟡 In Progress**
- CUDA runtime wired: context management, device enumeration, compute capability detection
- `InferenceEngine` with CUDA integration (`gpu_available()`, `full_device_info()`)
- KV cache, TMA descriptor binding, TMA bridge implemented
- GEMM and attention GPU kernels: stubbed (cpu path works)
- **Blocker**: 77 compilation errors (cuda-oxide trait bounds, type mismatches, missing gguf functions)

**Phase 3 (Production): 🔴 Not Started**
- Runner bridge stub exists, streaming unimplemented
- Safetensors weight loading available, GGUF file writer missing
- HuggingFace model download integration (hf-hub present, unused)

See `ROADMAP.md` for the detailed implementation plan.

## License

MIT OR Apache-2.0 (except `gguf/` which is AGPL-3.0-or-later)
