# LLM-Workspace Roadmap

## Status Overview

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 1: CPU Inference** | ✅ Complete | Full transformer + llama.cpp FFI path |
| **Phase 1.5: Hybrid Routing** | ✅ Complete | GPU → Remote → CPU device selector |
| **Phase 2: GPU Acceleration** | 🟡 In Progress | CUDA context wired, kernels stubbed |
| **Phase 3: Production** | 🔴 Not Started | Multi-model, streaming, model download |
| **Phase 4: Advanced Kernels** | 🔮 Future | FlashAttention, speculative decoding |

---

## Phase 1: CPU Inference (✅ Complete)

**Goal:** Run a real llama-style model on CPU using existing GGUF weights.

### Pure Rust Transformer Path (`llm-runner/src/transformer/`)

- [x] Wire `load_gguf_weights()` output to `LlamaModel` — map tensor names to model layers
- [x] Implement Q/K/V linear projections
- [x] Implement multi-head attention (CPU path)
- [x] Implement FFN layers with SwiGLU activation
- [x] Implement RMSNorm (layer normalization)
- [x] Implement RoPE positional embeddings
- [x] Wire tokenizer to GGUF vocab or fast tokenizer
- [x] Wire token sampling to model (temp, top-p, top-k)
- [x] Implement LM head (final linear layer for logits)
- [x] Connect `LlamaModel::forward()` to actual weight data end-to-end
- [x] Architecture-aware weight loading (llama, mistral, gemma, qwen2, qwen3, phi3, mixtral, starcoder2)
- [x] `LlamaModel::generate()` — autoregressive generation loop

### llama.cpp FFI Path (`llm-runner/src/llama/`)

- [x] `LlamaRunner` — high-level wrapper over llama-cpp-2
- [x] `LlamaRunnerBuilder` — builder pattern for context/model config
- [x] Tokenization / detokenization
- [x] Full generation loop with timing (prompt eval, token eval)
- [x] Chat template application (system, user, assistant)
- [x] Sampling with configurable parameters (temperature, top-k, top-p, min-p, TFS, typical p, repetition penalty)
- [x] Grammar-constrained decoding (JSON schema → grammar)
- [x] Session save/load
- [x] Embeddings
- [x] KV cache management and reset
- [x] Memory breakdown inspection
- [x] Model info extraction (params, embedding dim, layers, heads, context, vocab)

### GGUF Weight Loading (`llm-runner/src/gguf_weight_loader.rs`)

- [x] F32 passthrough
- [x] F16 / BF16 → f32 conversion
- [x] Q4_0 dequantization
- [x] Q4_1 dequantization
- [x] Q8_0 dequantization
- [x] Q2_K dequantization
- [x] Q3_K dequantization
- [x] Q4_K / Q4_K_M dequantization
- [x] Q5_K / Q5_K_M / Q5_K_S dequantization
- [x] Q6_K / Q6_K_S dequantization
- [x] Q8_K / Q8_K_M dequantization
- [x] Q1_K dequantization
- [x] I8 / I16 / I32 / I64 passthrough

---

## Phase 1.5: Hybrid Device Routing (✅ Complete)

**Goal:** Multi-device inference with local GPU + remote LM Studio discovery.

- [x] `device_discovery.rs` — enumerate local CUDA GPUs with VRAM info
- [x] `remote_discovery.rs` — remote LM Studio health checks via HTTP
- [x] `DeviceSelector` — priority-based device routing (GPU → Remote → CPU)
- [x] `RunnerBridge::send_remote_request()` — HTTP transport to remote LM Studio
- [x] `RunnerBridge::parse_lm_studio_response()` — LM Studio response parsing
- [x] `DeviceRouter` — combines DeviceSelector + RunnerBridge into execution pipeline
- [x] `ModelManager` — popularity scoring, smart preloading, background task
- [x] `Registry` — in-memory HashMap + filesystem auto-discovery
- [x] `DeviceInfo` / `DeviceSelection` — device info and selection result types

---

## Phase 2: GPU Acceleration (🟡 In Progress)

**Goal:** Replace CPU kernels with tcgen05 (sm_120) GPU kernels.

**Status:** CUDA runtime wired, PTX loading functional, GPU path available when CUDA present.

### Completed

- [x] CUDA context management (`cuda_runtime.rs`) — `CudaRuntime`, device enumeration, compute capability detection
- [x] `DeviceBuffer` extended with `Cuda` variant
- [x] `KernelFromPtx` wired to load PTX via `cuda-core`
- [x] `InferenceEngine` updated with CUDA runtime — `gpu_available()`, `full_device_info()`, `list_devices()`
- [x] `CudaDeviceInfo::supports_tcgen05()` / `supports_wgmma()` — compute capability checks
- [x] KV cache (`kernel/kvcache.rs`) — per-layer key/value caches with append
- [x] TMA descriptor binding (`kernel/tma_descriptor.rs`)
- [x] TMA bridge (`kernel/tma_bridge.rs`) — TMA descriptor → device buffer mapping

### Remaining

- [ ] Fix compilation errors (77 errors) — cuda-oxide trait bounds, type mismatches, missing gguf functions
- [ ] Implement real tcgen05 WGMMA matmul kernel (replace stub in `KernelFromPtx.matmul`)
- [ ] Implement GPU attention kernel with TMA descriptor binding
- [ ] Implement device memory allocation (`cuMemAlloc`/`cuMemFree`) via cuda-core `memory` module
- [ ] Implement H2D/D2H async memory transfers (`cuMemcpyHtoDAsync`) via cuda-core `memory` module
- [ ] Add CUDA error handling and CPU fallback logic
- [ ] Implement WGMMA for sm_120 (RTX 5060 Ti / 5090)
- [ ] Add fp8 support (cuda-device `f8` feature already in workspace)
- [ ] CPU attention optimization (candle-core or flash attention)

---

## Phase 3: Production Readiness (🔴 Not Started)

**Goal:** Multi-model support, streaming, and runner bridge.

- [ ] Implement runner bridge (HTTP or local pipe transport) — `RunnerBridge::send_request()` stub
- [ ] K-family dequantization verification and tests (Q2_K through Q8_K) — code exists, unverified against real models
- [ ] Model architecture routing (llama/mistral/qwen/gemma weight name mapping) — ✅ already done
- [ ] Streaming/lazy tensor loading for large models
- [ ] Weight loading from safetensors (currently only GGUF path)
- [ ] Memory management and model unloading
- [ ] Implement GGUF file writer (currently parser-only)
- [ ] Implement safetensors file writer (currently parser-only)
- [ ] HuggingFace model download integration (hf-hub dependency present but unused)

---

## Phase 4: Advanced Kernels (🔮 Future)

- [ ] WGMMA for sm_100 (Blackwell) — separate from sm_120 path
- [ ] FlashAttention-2 style fused kernels
- [ ] Speculative decoding support
- [ ] Multi-GPU tensor parallelism

---

## Current Architecture

```
llm-workspace/
├── gguf/                    GGUF parser (working, all 29+ quant types)
├── gguf-cli/                CLI inspector (working)
├── safetensors/             SQLite-backed weight storage, safetensors parser (working)
├── llm-plug-in/             Protocol + templates (working)
├── llm-runner/              Inference engine
│   ├── transformer/         Pure-Rust LlamaModel: Q/K/V, attention, FFN, RMSNorm, RoPE, sampling, tokenizer ✅
│   ├── llama/               llama.cpp FFI: LlamaRunner with full generation, chat, embeddings, grammar ✅
│   ├── gguf_weight_loader.rs  K-family dequantization (Q1_K through Q8_K_M) ✅
│   ├── inference_engine.rs  GEMM + attention kernels, CUDA integration 🟡
│   ├── model.rs             Model with KV cache, prefill/decode loop
│   ├── cuda_runtime.rs      CUDA context, device enumeration 🟡
│   ├── device.rs            DeviceSelector + DeviceRouter (GPU→Remote→CPU) ✅
│   ├── device_discovery.rs  Local GPU enumeration ✅
│   ├── remote_discovery.rs  Remote LM Studio health checks ✅
│   ├── runner.rs            RunnerBridge + DeviceRouter ✅
│   ├── model_manager.rs     Popularity scoring, smart preloading ✅
│   ├── registry.rs          Model discovery (in-memory + filesystem) ✅
│   ├── kernel/              Device buffers, TMA, KV cache
│   │   ├── gemm.rs          CPU GEMM working, GPU stubbed
│   │   ├── attention.rs     CPU attention working, GPU stubbed
│   │   ├── kvcache.rs       Per-layer KV cache ✅
│   │   ├── tma_bridge.rs    TMA descriptor → device buffer ✅
│   │   └── tma_descriptor.rs TMA descriptor binding ✅
│   └── model_loader.rs      Safetensors-backed weight loading
├── cuda-oxide/              Host/device crates (added, not published)
└── rust-toolchain.toml      Pinned nightly (required for cuda-oxide)
```

## Hardware Topology

| Device | VRAM | Free | Role |
|--------|------|------|------|
| GPU0 (RTX 4070) | 16GiB | ~1.6GiB | LM Studio (~13GiB) |
| GPU1 (RTX 5060 Ti) | 16GiB | ~3.6GiB | LM Studio (~12.6GiB) |
| Remote 3070 Ti | 8GiB | ~8GiB | LM Studio on Tailscale |
| Laptop (LM-Link) | unknown | unknown | LM Studio on Tailscale |

## Key Dependencies

- **llama-cpp-2** — llama.cpp Rust bindings (FFI path)
- **cuda-oxide** — `cuda-core`, `cuda-device`, `cuda-host`, `cuda-macros`, `cuda-bindings`, `cuda-async`, `libnvvm-sys`, `nvjitlink-sys`, `reserved-oxide-symbols`
- **candle-core/nn/transformers** — ML inference backbone (pure-Rust path)
- **half** — f16/f32/f8 types
- **gguf parser** — self-hosted, all 29+ quantization types
- **safetensors crate** — safe model weight deserialization
- **rusqlite** — SQLite for safetensors storage

## Notes

- `rustc-codegen-cuda` is intentionally excluded — requires `#![feature(rustc_private)]` and is a dylib rustc codegen backend, not a regular dependency
- Bare `nightly` toolchain was corrupted (rustc out of sync with rustlib) — pinned to `nightly-2026-05-06` initially, now reinstalled and working
- K-family dequantization tests are marked `#[ignore]` — code exists but unverified against real models
- **Current blocker**: 77 compilation errors in llm-runner (cuda-oxide trait bounds, type mismatches, missing gguf functions in llama-cpp-sys-2)
