# LLM-Workspace Roadmap

## Status: ~80% — Transformer layers wired, tokenizer and sampling done, hybrid device routing added

### Phase 1: CPU Inference (~2-3 weeks)

**Goal:** Run a real llama-style model on CPU using existing GGUF weights.

- [x] Wire `load_gguf_weights()` output to `Model` — map tensor names to model layers
- [x] Implement Q/K/V linear projections
- [x] Implement multi-head attention (CPU path)
- [x] Implement FFN layers with SwiGLU activation
- [x] Implement RMSNorm (layer normalization)
- [x] Implement RoPE positional embeddings
- [x] Wire tokenizer to GGUF vocab or fast tokenizer
- [x] Wire token sampling to model (temp, top-p, top-k) — `sample()` exists in `transformer/sampling.rs`, wired into `LlamaModel::generate()`
- [x] Implement LM head (final linear layer for logits)
- [x] Connect `Model::run()` to actual weight data end-to-end — added `CpuModel` that bridges `LlamaModel` weights into the prefill/decode loop; `Model` remains GPU-focused for future kernel work

### Phase 1.5: Hybrid Device Routing (new)

**Goal:** Multi-device inference with local GPU + remote LM Studio discovery.

- [x] `device_discovery.rs` — enumerate local CUDA GPUs with VRAM info (stubbed compute cap)
- [x] `remote_discovery.rs` — Tailscale LM Studio network discovery
- [x] `DeviceSelector` — priority-based device routing (GPU → Remote → CPU)
- [x] `RunnerBridge::send_remote_request()` — HTTP transport to remote LM Studio
- [x] `RunnerConfig` — extended with `remote_endpoints` and `device_priority`
- [x] Wire `DeviceSelector` into inference pipeline
- [ ] Add CLI `devices list` and `devices route` commands
- [ ] Model size-based auto-routing (small models → GPU, large → remote)
- [ ] Health check polling for remote devices

### Phase 2: GPU Acceleration (~3-4 weeks)

**Goal:** Replace CPU kernels with tcgen05 (sm_120) GPU kernels.

**Status:** CUDA runtime wired, PTX loading functional, GPU path available when CUDA present.

- [x] CUDA context management (`cuda_runtime.rs`) — `CudaRuntime`, device enumeration, compute capability detection
- [x] `DeviceBuffer` extended with `Cuda` variant — real cuda-core DeviceBuffer integration
- [x] `KernelFromPtx` wired to load PTX via `cuda-core` — `load_module_from_ptx_src`, kernel lookup, launch
- [x] `InferenceEngine` updated with CUDA runtime — `gpu_available()`, `full_device_info()`, `list_devices()`
- [ ] Implement real tcgen05 WGMMA matmul kernel (replace stub in `KernelFromPtx.matmul`)
- [ ] Implement GPU attention kernel with TMA descriptor binding
- [ ] Implement device memory allocation (`cuMemAlloc`/`cuMemFree`) via cuda-core `memory` module
- [ ] Implement H2D/D2H async memory transfers (`cuMemcpyHtoDAsync`) via cuda-core `memory` module
- [ ] Add CUDA error handling and CPU fallback logic
- [ ] Implement WGMMA for sm_120 (RTX 5060 Ti / 5090)
- [ ] Add fp8 support (cuda-device `f8` feature already in workspace)

### Phase 3: Production Readiness (~2-3 weeks)

**Goal:** Multi-model support, streaming, and runner bridge.

- [ ] Implement runner bridge (HTTP or local pipe transport)
- [ ] K-family dequantization verification and tests (Q2_K through Q8_K)
- [ ] Model architecture routing (llama/mistral/qwen/gemma weight name mapping)
- [ ] Streaming/lazy tensor loading for large models
- [ ] Weight loading from safetensors (currently only GGUF path)
- [ ] Memory management and model unloading
- [ ] Implement GGUF file writer (currently parser-only)
- [ ] Implement safetensors file writer (currently parser-only)
- [ ] HuggingFace model download integration (hf-hub dependency present but unused)

### Phase 4: Advanced Kernels (future)

- [ ] WGMMA for sm_100 (Blackwell) — separate from sm_120 path
- [ ] FlashAttention-2 style fused kernels
- [ ] Speculative decoding support
- [ ] Multi-GPU tensor parallelism

---

## Current Architecture

```
llm-workspace/
├── gguf/                    GGUF parser (working)
├── gguf-cli/                CLI inspector (working)
├── safetensors/             Weight storage + parsing (working)
├── llm-plug-in/             Protocol + templates (working)
├── llm-runner/              Inference engine (stubbed)
│   ├── kernel/              Device buffers, TMA, kv cache (working)
│   ├── kernel/gemm.rs       CPU GEMM working, GPU stubbed
│   ├── kernel/attention.rs  CPU attention working, GPU stubbed
│   ├── model.rs             Model struct (stubbed)
│   ├── model_loader.rs      Weight loading (partial)
│   ├── registry.rs          Model discovery (working)
│   └── tokenizer.rs         Stubbed
├── cuda-oxide/              Host/device crates (added)
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

- **cuda-oxide** — `cuda-core`, `cuda-device`, `cuda-host`, `cuda-macros`, `cuda-bindings`, `cuda-async`, `libnvvm-sys`, `nvjitlink-sys`, `reserved-oxide-symbols`
- **candle-core/nn/transformers** — ML inference backbone
- **half** — f16/f32/f8 types
- **gguf parser** — self-hosted, all 29+ quantization types
- **safetensors crate** — safe model weight deserialization

## Notes

- `rustc-codegen-cuda` is intentionally excluded — requires `#![feature(rustc_private)]` and is a dylib rustc codegen backend, not a regular dependency
- Bare `nightly` toolchain was corrupted (rustc out of sync with rustlib) — pinned to `nightly-2026-05-06` initially, now reinstalled and working
- All LLVM libraries had missing execute permissions — fixed on all toolchains
- K-family dequantization tests are marked `#[ignore]` — code exists but unverified against real models
