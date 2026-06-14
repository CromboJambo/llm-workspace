# PESTI Roadmap

## Status Overview

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1: CPU Inference** | ✅ Complete | Pure-Rust transformer + llama.cpp FFI path |
| **Phase 1.5: Hybrid Routing** | ✅ Complete | GPU → Remote → CPU device selector |
| **Phase 2: Backend Abstraction** | 🟡 In Progress | Trait layer, tensor interfaces, execution dispatch, error handling |
| **Phase 3: Runtime** | 🔴 Not Started | Runner bridge, streaming, model management |
| **Phase 4: GPU Kernels** | 🔮 Future | WGMMA, fp8, attention (after abstraction is solid) |

---

## Phase 1: CPU Inference (✅ Complete)

**Goal:** Run a real llama-style model on CPU using existing GGUF weights.

### Pure Rust Transformer Path (`llm-runner/src/transformer/`)

- [x] Wire `load_gguf_weights()` output to `LlamaModel`
- [x] Q/K/V linear projections
- [x] Multi-head attention (CPU path)
- [x] FFN layers with SwiGLU activation
- [x] RMSNorm, RoPE positional embeddings
- [x] Architecture-aware weight loading (llama, mistral, gemma, qwen2, qwen3, phi3, mixtral, starcoder2)
- [x] `LlamaModel::generate()` — autoregressive generation loop
- [x] Tokenizer wiring, token sampling (temp, top-p, top-k)
- [x] LM head, logit computation

### llama.cpp FFI Path (`llm-runner/src/llama/`)

- [x] `LlamaRunner` + builder pattern
- [x] Tokenization / detokenization
- [x] Full generation loop with timing
- [x] Chat templates, grammar-constrained decoding
- [x] Session save/load, embeddings
- [x] Configurable sampling (top-k, top-p, min-p, TFS, typical p, repetition penalty)
- [x] KV cache management, memory inspection, model info extraction

### GGUF Weight Loading (`llm-runner/src/gguf_weight_loader.rs`)

- [x] All 29+ quantization types: Q1_K through Q8_K_M, F32/F16/BF16, I8/I16/I32/I64

---

## Phase 1.5: Hybrid Device Routing (✅ Complete)

**Goal:** Multi-device inference with local GPU + remote LM Studio discovery.

- [x] `device_discovery.rs` — local CUDA GPU enumeration with VRAM info
- [x] `remote_discovery.rs` — remote LM Studio health checks via HTTP
- [x] `DeviceSelector` — priority-based routing (GPU → Remote → CPU)
- [x] `RunnerBridge` — HTTP transport to remote LM Studio
- [x] `DeviceRouter` — combines discovery + transport into execution pipeline
- [x] `ModelManager` — popularity scoring, smart preloading
- [x] `Registry` — in-memory + filesystem model discovery

---

## Phase 2: Backend Abstraction Layer (🟡 In Progress)

**Goal:** Define the execution trait layer so CUDA is one backend among others, not the center.

### Completed

- [x] CUDA runtime wired: context management, device enumeration, compute capability detection
- [x] `DeviceBuffer` with `Cuda` variant
- [x] `KernelFromPtx` — PTX loading via cuda-core
- [x] `InferenceEngine` with CUDA integration (`gpu_available()`, `full_device_info()`)
- [x] `CudaDeviceInfo::supports_tcgen05()` / `supports_wgmma()` — compute capability checks
- [x] KV cache (`kernel/kvcache.rs`) — per-layer key/value caches with append
- [x] TMA descriptor binding (`kernel/tma_descriptor.rs`)
- [x] TMA bridge (`kernel/tma_bridge.rs`) — descriptor → device buffer mapping
- [x] Fixed compilation errors (77 → 0 errors)
- [x] **Error handling overhaul:**
  - `RunnerError::Cuda` — proper `CudaError` → `RunnerError` conversion via `#[from]`
  - `RunnerError::Gemm` — structured GEMM errors with arch, m/n/k dimensions
  - `RunnerError::Attention` — structured attention errors with num_heads, head_dim, seq
  - **Runtime CPU fallback** — `InferenceEngine::matmul()` and `attention()` automatically retry on CPU when GPU fails
  - `DeviceBackend::is_available()` — fixed inverted logic (was returning false for CUDA)
  - `CudaAttentionKernel::is_available()` — now checks both arch AND CUDA driver availability
  - `CudaAttentionKernel::forward()` — validates buffer backing, returns properly-sized output

### Key Design Decisions

- **CUDA is a backend, not the substrate.** The tensor interfaces (`GemmKernel`, `AttentionKernel`) define the contract; cuda-oxide implements one path.
- **TMA descriptors are speculative.** The CUDA driver API treats `CUtensorMap` as opaque. Production descriptors should use `cuTensorMapEncodeTiled()` on the host. `HostTmaDescriptor` wraps the correct approach.
- **CPU is the default path.** GPU is an optimization, not a requirement. All CPU paths are verified and working.

### Remaining

- [ ] **Dispatch logic** — `DeviceRouter` → backend selector that routes tensor ops to the right impl
- [ ] **Async memory transfers** — H2D/D2H via cuda-core `memory` module (partially done: `memcpy_htod_async` etc. exist in `CudaMemoryBackend`)

### Why This Before Kernels

Real GPU kernels (WGMMA, fp8, etc.) are high-effort, hardware-specific work. Without a clean abstraction layer, that effort is locked to one backend. Get the trait layer right first, and kernels become interchangeable implementations.

---

## Phase 3: Runtime (🔴 Not Started)

**Goal:** Make the runner usable as a library and service.

- [ ] **Runner bridge** — HTTP or local pipe transport for remote inference
- [ ] **Streaming token generation** — progressive output, not batch
- [ ] **Model lifecycle** — loading, unloading, memory management, popularity-based eviction
- [ ] **SafeTensors weight loading** — wire `ModelLoader` for SafeTensors → `LlamaModel`
- [ ] **GGUF file writer** — currently parser-only
- [ ] **SafeTensors file writer** — currently parser-only
- [ ] **HuggingFace model download** — integrate `hf-hub` dependency

---

## Phase 4: GPU Kernels (🔮 Future)

**Goal:** Replace CPU kernels with hardware-accelerated implementations behind the abstraction layer.

- [ ] WGMMA matmul for sm_120 (RTX 5060 Ti) via PTX
- [ ] WGMMA for sm_100 (Blackwell) — separate code path
- [ ] GPU attention kernel with TMA descriptor binding
- [ ] fp8 support (cuda-device `f8` feature already in workspace)
- [ ] CPU attention optimization (candle-core or flash attention)
- [ ] Multi-GPU tensor parallelism (long-term)

---

## Near-Term Priorities (Next 2-4 Weeks)

### 1. Dispatch Logic (Highest Impact)

Wire `DeviceRouter` → backend selector. When a tensor op is requested, the router picks the right backend based on device availability and tensor layout. The trait layer is solid (GemmKernel, AttentionKernel, MemoryBackend all implemented), but nothing routes ops through them yet.

### 2. K-Family Dequantization Verification

Test all Q2_K through Q8_K quant types against real GGUF models. Remove `#[ignore]` from tests once verified.

### 3. SafeTensors Weight Loading

Wire `ModelLoader` to load SafeTensors → `LlamaModel`. Enable loading converted GGUF→SafeTensors weights.

---

## Architecture

```
pesti/
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
├── cuda-oxide/              Host/device crates (one backend)
└── rust-toolchain.toml      Pinned nightly
```

## Key Dependencies

- **llama-cpp-2** — llama.cpp Rust bindings (FFI path)
- **cuda-oxide** — `cuda-core`, `cuda-device`, `cuda-host`, `cuda-macros`, `cuda-bindings`, `cuda-async`, `libnvvm-sys`, `nvjitlink-sys`, `reserved-oxide-symbols`
- **candle-core/nn/transformers** — ML inference backbone (pure-Rust path)
- **half** — f16/f32/f8 types
- **gguf parser** — self-hosted, all 29+ quantization types
- **safetensors crate** — safe model weight deserialization
- **rusqlite** — SQLite for safetensors storage

## Notes

- `rustc-codegen-cuda` is intentionally excluded — requires `#![feature(rustc_private)]` and is a dylib rustc codegen backend
- Nightly toolchain: pinned to working version
- K-family dequantization tests are marked `#[ignore]` — code exists but unverified against real models
- **TMA descriptor is speculative** — use `HostTmaDescriptor` + `cuTensorMapEncodeTiled` for production
- CUDA is one backend, not the center. The abstraction layer (Phase 2) determines what the rest of the stack needs
