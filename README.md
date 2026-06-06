# llm-workspace

LLM inference stack: runner, plug-in protocol, safetensors, GGUF parser.

## Workspace Members

| Crate | Description | Status |
|-------|-------------|--------|
| `gguf` | GGUF model weight file parser: header, tensor metadata, KV config, quantization type detection | Working |
| `gguf-cli` | CLI tool to inspect GGUF model files | Working |
| `safetensors` | SQLite-backed weight storage, safetensors parser, GGUF-to-safetensors conversion | Working |
| `llm-plug-in` | Weight manifest generation, inference request/response protocol, prompt templates | Working |
| `llm-runner` | Inference engine: tensor computation, model loading, tokenizer, device backend, hybrid routing | Partial |

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

GGUF parsing, safetensors storage, and weight loading are functional. The transformer model architecture (`LlamaModel`) is implemented: Q/K/V projections, multi-head attention (CPU), FFN with SwiGLU, RMSNorm, RoPE, LM head, and end-to-end `forward()` all wired. GPU kernel path (tcgen05/WGMMA) is stubbed. **New: hybrid device routing** — `DeviceSelector` with priority-based routing (local GPU → remote LM Studio → CPU), `RemoteDevice` discovery via Tailscale network, and `RunnerBridge::send_remote_request()` for HTTP transport to remote LM Studio. Device discovery uses cudarc driver API with stubbed compute capability (Phase 2). Remaining: tokenizer wiring, sampling integration, GPU kernel implementation, and model size-based auto-routing. See `ROADMAP.md` for the implementation plan.

## License

MIT OR Apache-2.0 (except `gguf/` which is AGPL-3.0-or-later)
