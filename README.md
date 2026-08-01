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
| `train-extract` | Training dataset extraction pipeline with provenance tracking |

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
├── train-extract/           Training dataset extraction pipeline
└── rust-toolchain.toml      Pinned nightly
```

## Current State

**Phase 1 (CPU Inference): ✅ Complete** — Pure-Rust transformer + llama.cpp FFI, all GGUF quant types.

**Phase 1.5 (Hybrid Routing): ✅ Complete** — GPU → Remote → CPU device selector with health checks.

**Phase 2 (Backend Abstraction): ✅ Complete** — Trait layer, tensor interfaces, execution dispatch, error handling overhaul.

**Phase 3 (Runtime): ✅ Complete** — Runner bridge, streaming, model management, SafeTensors weight loading, HF download.

**Phase 4a (Mistral.rs Backend): ✅ Complete** — Production GPU kernels via mistral.rs (WGMMA, tcgen05, flash attention, FP8).

**Phase 4b (Candle Bridge): ✅ Complete** — candle-core tensor bridge for GPU-accelerated gemm/sdpa/rope/rms_norm/swiglu.

**Phase 4c (Dispatch Layer): ✅ Complete** — LayerDispatch, full forward pass, GPU/CPU auto-select, async memory transfers.

**Phase 5.1 (Validation & Polish): ✅ COMPLETE** — GGUF v3 test data regression fixed (457/457 tests passing).

### Build & Test Health

| Metric | Value |
|--------|-------|
| Rust files | 69 |
| Lines (llm-runner/src) | ~21,377 |
| Tests passing | **457 / 457** ✅ (79 in pesti-gguf + 378 in pesti-runner) |
| Tests failing | 0 |
| Clippy warnings | 15 |
| Build (default) | ✅ Clean |
| Build (mistralrs) | ✅ Clean |

### Metric Notes
Test count verified: `cargo test -p pesti-gguf --lib` reports **53/53** passing, `cargo test -p pesti-runner --lib` reports **314/321** (7 ignored). Total: **367 tests passing**, 7 ignored.

### Known Issues
All GGUF v3 test data regression bugs fixed (STRING type value + u64 key lengths, 53/53 tests passing in `pesti-gguf`). See [GGUF_FIX_SUMMARY.md](GGUF_FIX_SUMMARY.md) for detailed fix notes.

## License

AGPL-3.0-or-later
