```
▄▄▄▄▄▄▄▄▄▄     ▄▄▄▄▄▄▄▄▄         ▄▄▄▄▄     ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄
█████ ▀█████▄  █████ ▀████▄   ▄▓▓▓▓▀▓▓▓▓▄  ▐▓█▓█▀▓███▓▀▓▓█▓▌ █▓█▓█
▓███▓  ▐█▓██▓▌ ▓█▓█▓  ▐▓█▓▓▌ ▐▓▓▓▓▌ ▐▓▓▓▓▌ ▐▓▓▓▌ ▓▓▓▓▓ ▐▓▓▓▌ ▓▓▓▓▓
▓▓▓▓▓   ▓▓▓▓▓▓ ▓▓▓▓▓   ▒▓▒▒░ ▓▒▓▒▒   ▒▒▓▒▓ ▀     ▒▓▒▓▓     ▀ ▓▓▓▒▓
▓▓▓▒▓   ▒▒▒▒▒▒ ▒▓▓▒▓         ▓▒▓▓▒▄              ▒▒▒▒▓       ▓▒▓▓▒
▒▓▒▓▒ ▄▒▒▒▒▒▒▌ ░▓▒▓▒▓▒       ▀▀▀▓▒▓▒░▒▒▀▄        ▒░▒▓▒       ░▒░▓▒
▓░▓▒▒▓▒▒▒░▒░▀  ░░▒░░   ░░░░░        ▀░▒░▒▒       ░░░░░       ░░░░░
░▀█░▀          ▀░▀░░   ▀░▀░▀ ░░▀░░   ▀░░░▄       ░░░▀░       ▀░▀░░
  ▀               ▀     ▀     ▀  ▀   ▒▀ ▀░       ▀ ▀          ▀  ▀
▓ ▓▄▀          ▄▓ ▄▓   ▓ ▄▓▄ ▓▄ ▓▄   ▓▄▄▓▓       █ ▄▓        ▓▄ ▓▄
▒▄▒▒▒          ▒▒▄░▒  ▓▒▒▒▒▓ ▄▒▓▒▄  ▄▄▒▒▒▒       ▓▓▓▒▒       ▄▒▓▒▄
░░░░░          ░░░▒░▄░░░░░░▀ ▀▄░░░░░░░░░▄▀      ▄▒▒▒░░▄      ░░░░░
```
Portable Execution Substrate for Transformer Inference.

A backend-agnostic Rust inference runtime with clean GGUF, SafeTensors, and execution abstractions.

## What This Is

PESTI is an inference runtime that separates model representation from execution. It provides:

- **Format layers** — GGUF parser (all 29+ quantization types) and SafeTensors storage
- **Execution paths** — pure-Rust CPU transformer + llama.cpp FFI wrapper
- **Device routing** — priority-based GPU → remote → CPU dispatch
- **Backend abstraction** — CUDA as one backend among others, not the center

The lasting contribution is the runtime, the abstractions, and the tensor interfaces — not any specific model.

## Workspace Members

| Crate | Description |
|-------|-------------|
| `gguf` | GGUF model weight file parser: header, tensor metadata, KV config, quantization types |
| `gguf-cli` | CLI tool to inspect GGUF model files |
| `safetensors` | SQLite-backed weight storage, SafeTensors parser, GGUF-to-SafeTensors conversion |
| `llm-plug-in` | Weight manifest generation, inference protocol, prompt templates |
| `llm-runner` | Inference engine with CPU transformer + llama.cpp FFI + device routing |

## Inference Paths

### Pure-Rust Transformer (CPU)

Full Llama-style model in pure Rust:
- Q/K/V projections, multi-head attention, FFN with SwiGLU
- RMSNorm, RoPE positional embeddings
- LM head, token sampling (temperature, top-p, top-k)
- Architecture-aware weight loading (llama, mistral, gemma, qwen2, phi3, mixtral, starcoder2)
- `LlamaModel::generate()` — autoregressive generation loop

### llama.cpp FFI

High-level wrapper over llama-cpp-2:
- `LlamaRunner` with builder pattern for context/model config
- Full generation with timing, chat templates, grammar-constrained decoding
- Session save/load, embeddings, configurable sampling

## Device Routing

`DeviceRouter` combines discovery with priority-based routing:

1. **Local GPU** — CUDA via cuda-oxide (stubbed kernels, CPU fallback)
2. **Remote LM Studio** — HTTP transport via `RunnerBridge` (health-checked)
3. **CPU** — fallback

## Requirements

- Rust nightly (pinned in `rust-toolchain.toml`)
- CUDA Toolkit (optional; CPU inference works without it)
- llama.cpp (via llama-cpp-2 crate; FFI path)

## Building

```bash
cargo check --workspace
cargo build -p pesti-runner
cargo test --workspace
```

## GGUF CLI

```bash
cargo run -p pesti-gguf-cli -- inspect <file.gguf>
cargo run -p pesti-gguf-cli -- list <file.gguf>
cargo run -p pesti-gguf-cli -- tensor <file.gguf> -t
cargo run -p pesti-gguf-cli -- tensor <file.gguf> -e <name>
```

## Architecture

```
llm-workspace/
├── gguf/                    GGUF parser (all 29+ quant types)
├── gguf-cli/                CLI inspector
├── safetensors/             SQLite-backed weight storage, SafeTensors parser
├── llm-plug-in/             Protocol + templates
├── llm-runner/              Inference engine
│   ├── transformer/         Pure-Rust LlamaModel ✅
│   ├── llama/               llama.cpp FFI ✅
│   ├── device.rs            DeviceSelector + DeviceRouter ✅
│   ├── device_discovery.rs  Local GPU enumeration ✅
│   ├── remote_discovery.rs  Remote LM Studio health checks ✅
│   ├── runner.rs            RunnerBridge + DeviceRouter ✅
│   ├── model_manager.rs     Popularity scoring, smart preloading ✅
│   ├── registry.rs          Model discovery ✅
│   ├── kernel/              Buffers, TMA, KV cache
│   │   ├── gemm.rs          CPU GEMM working, GPU stubbed
│   │   ├── attention.rs     CPU attention working, GPU stubbed
│   │   ├── kvcache.rs       Per-layer KV cache ✅
│   │   ├── tma_bridge.rs    TMA descriptor → device buffer ✅
│   │   └── tma_descriptor.rs TMA binding (SPECULATIVE) ⚠️
│   └── model_loader.rs      SafeTensors weight loading
├── cuda-oxide/              CUDA host/device crates (one backend)
└── rust-toolchain.toml      Pinned nightly
```

## Current State

**Phase 1 (CPU Inference): ✅ Complete** — Pure-Rust transformer + llama.cpp FFI, all GGUF quant types.

**Phase 1.5 (Hybrid Routing): ✅ Complete** — GPU → Remote → CPU device selector with health checks.

**Phase 2 (GPU Acceleration): 🟡 In Progress** — CUDA runtime wired, kernels stubbed. See `ROADMAP.md`.

**Phase 3 (Production): 🔴 Not Started** — Runner bridge, streaming, model download.

## License

MIT OR Apache-2.0 (except `gguf/` which is AGPL-3.0-or-later)
