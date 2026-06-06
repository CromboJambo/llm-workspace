# LLM-Workspace Roadmap

## Status: ~75% — Transformer layers wired, tokenizer and sampling remain

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

### Phase 2: GPU Acceleration (~3-4 weeks)

**Goal:** Replace CPU kernels with tcgen05 (sm_120) GPU kernels.

- [ ] Merge tcgen05 example into llm-runner crate
- [ ] Wire `KernelFromPtx` to actual PTX loading via `cuda-core`
- [ ] Implement real tcgen05 WGMMA matmul kernel
- [ ] Implement GPU attention kernel with TMA descriptor binding
- [ ] Implement device memory allocation (`cuMemAlloc`/`cuMemFree`)
- [ ] Implement H2D/D2H async memory transfers (`cuMemcpyHtoDAsync`)
- [ ] Add CUDA context management (`cuCtxCreate`/`cuCtxSetCurrent`)
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

Thought: 3.0s
The user asked me to add a roadmap to the project. I've created a ROADMAP.md file. The user didn't ask me to commit it, so I should just stop here.
Done. ROADMAP.md created at /home/crombo/projects/llm-workspace/ROADMAP.md with 4 phases:
1. CPU Inference — wire weights to Model, implement transformer layers, RoPE, tokenizer, sampling
2. GPU Acceleration — merge tcgen05 example, wire cuda-core, implement H2D/D2H transfers, GPU attention
3. Production Readiness — runner bridge, K-family dequantization, model routing, lazy loading
4. Advanced Kernels — WGMMA sm_100, FlashAttention-2, speculative decoding, tensor parallelism
