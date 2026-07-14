# PESTI Roadmap - Updated with Research Integration Patterns

**Last updated:** 2026-07-13  
**Source:** Post-lumen/pon research synthesis + current project state

---

## Status Overview

| Phase | Status | Focus | Confidence |
|-------|--------|-------|------------|
| **Phase 1: CPU Inference** | ✅ Complete | Pure-Rust transformer + llama.cpp FFI path | High |
| **Phase 1.5: Hybrid Routing** | ✅ Complete | GPU → Remote → CPU device selector | High |
| **Phase 2: Backend Abstraction** | ✅ Complete | Trait layer, tensor interfaces, execution dispatch | High |
| **Phase 3: Runtime** | ✅ Complete | Runner bridge, streaming, model management, SafeTensors, HF download | High |
| **Phase 4a: Mistral.rs Backend** | ✅ Complete | Production GPU kernels via mistral.rs (WGMMA, tcgen05, flash attention) | High |
| **Phase 4b: Candle Bridge** | ✅ Complete | candle-core tensor bridge for GPU-accelerated operations | High |
| **Phase 4c: Dispatch Layer** | ✅ Complete | LayerDispatch, full forward pass, GPU/CPU auto-select | High |
| **Phase 5.1: Validation & Polish** | 🔮 Next Sprint | Fix GGUF v3 test data regression (77/79 tests passing) | Medium-High |
| **Phase 5.2: Differential Conformance Testing** | 🎯 Research-Driven | Byte-exact comparison against llama.cpp/candle-core references | Low-Medium* |
| **Phase 5.3: Tiered Execution with Tier-Up Thresholds** | 🔮 Q4 2026 | Profile-driven optimization paths (lumen/pon pattern) | Medium-Low* |
| **Phase 6: Model IR Abstraction** | 🎯 Early 2027 | Single IR feeding multiple backends (pon pattern) | Low-Medium* |

\* Confidence levels based on research insights from lumen & pon projects

---

## Phase 5.1: Validation & Polish (Next Sprint) 🔴 Critical Path

### Goal
Fix GGUF v3 test data regression and establish baseline conformance testing infrastructure.

### Current State
- **Test Coverage:** 77/79 tests passing (2 failures in near-duplicate detection feature, not related to GGUF v3)
- **Known Issue:** `kv_pair_u32()` helpers creating `Uint32` values instead of `Uint64` for v3 wire format

### Action Items

#### 1. Fix Test Data Helpers (PR #XXX)
**Files to modify:**
- `llm-runner/src/transformer/model.rs`: Line ~1269 - `kv_pair_u32` → `kv_pair_u64`
- `llm-runner/src/transformer/tokenizer.rs`: Line ~267 - same fix
- `llm-runner/src/model_loader.rs`: Line ~344 - same fix  
- `llm-runner/src/gguf_weight_loader.rs`: Line ~1182 - same fix
- `llm-runner/tests/dispatch_integration.rs`: Line ~19 - module-level helper fix

**Expected outcome:** 79/79 tests passing (all existing tests)

#### 2. Implement Basic Conformance Testing Skeleton
**New crate:** `llm-runner/crates/pesti-conformance`

**Structure:**
```
crates/pesti-conformance/
├── src/
│   ├── lib.rs           # Public API for conformance testing
│   ├── corpus.rs        # Corpus loading & management
│   ├── reference_llama_cpp.rs  # llama.cpp FFI integration
│   ├── reference_candle.rs     # candle-core GPU integration (future)
│   └── delta_minimizer.rs    # Divergence minimization logic
├── tests/
│   └── golden_models/   # Known-good model outputs for regression testing
└── Cargo.toml
```

**Key features:**
- Corpus directory: `/home/crombo/projects/llm-workspace/conformance-corpus/`
- Reference implementations: llama.cpp FFI (immediate), candle-core GPU (future)
- Ratcheted floor file: `conformance-floor.json` with minimum passing counts
- Delta minimization: Reduce divergences to minimal reproducible test cases

**Timeline:** 2-3 weeks for MVP

#### 3. Add Benchmarking Infrastructure
**Files to modify:**
- `crates/llm-runner/benches/conformance_bench.rs`: Compare pesti vs llama.cpp output quality
- `crates/llm-runner/benches/tier_transition_bench.rs`: Measure tier-up threshold overheads

---

## Phase 5.2: Tiered Execution with Profile-Driven Tier-Up (Q4 2026) 🟡 Medium Priority

### Goal
Implement profile-driven optimization paths inspired by lumen's tier-0→tier-1→tier-2 model.

### Current Gap
DeviceRouter exists as priority-based selector (GPU → Remote → CPU), but lacks runtime profiling to decide *when* a model "earns" GPU acceleration vs staying on CPU baseline.

### Design Sketch

```rust
// dispatch.rs - New tiered execution infrastructure
pub struct LayerDispatch {
    call_count: AtomicUsize,        // Track invocations per layer
    tier_threshold: usize,          // e.g., 100 calls before GPU optimization
    current_tier: Tier,             // Current execution tier (CPUBaseline | GpuOptimized)
}

pub enum Tier {
    CPUBaseline,         // Generic CPU kernels (correctness oracle)
    GpuFlashAttention,   // Flash attention + WGMMA optimizations
    GpuCudaOxide,        // Full cuda-oxide backend with kernel fusion
}

impl LayerDispatch {
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Increment call count and check if threshold exceeded
        let new_count = self.call_count.fetch_add(1, SeqCst);
        
        // Tier up when threshold exceeded (or immediately for hot paths)
        let target_tier = if new_count > self.tier_threshold || has_loop(input) {
            Tier::GpuFlashAttention
        } else {
            Tier::CPUBaseline
        };
        
        if target_tier != self.current_tier {
            debug!("Tier transition: {:?} → {:?}", self.current_tier, target_tier);
            self.current_tier = target_tier;
        }
        
        match self.current_tier {
            Tier::CPUBaseline => cpu_forward(input),
            Tier::GpuFlashAttention => gpu_flash_attention(input),
            Tier::GpuCudaOxide => cuda_oxide_forward(input),
        }
    }
}

// model.rs - Add tier-up flag to LlamaModel
pub struct LlamaModel {
    dispatch: Option<DispatchContext>,
    enable_tier_up: bool,              // Opt-in via Model::enable_tier_up()
    tier_threshold: usize,             // Default 100 invocations
}

impl LlamaModel {
    pub fn enable_tier_up(&mut self) {
        self.enable_tier_up = true;
        self.tier_threshold = 100; // Configurable threshold
    }
    
    pub fn forward_with_dispatch(&mut self, input: &Tensor) -> Result<Tensor> {
        if let Some(ref mut dispatch) = self.dispatch {
            dispatch.forward(input) // Triggers tier-up logic
        } else {
            cpu_forward(input)
        }
    }
}
```

### Implementation Phases

#### Phase A: Profiling Infrastructure (Weeks 1-2)
- Add call-count tracking to `DeviceRouter` and `LayerDispatch`
- Integrate with existing metrics system (prometheus/otel)
- Benchmark baseline performance vs tiered execution overhead

#### Phase B: Tier Transition Logic (Weeks 3-4)
- Implement tier-up threshold checks in dispatch layer
- Add loop detection heuristics (inspired by lumen's "immediate tier-up if body contains loop")
- Add configuration API (`LUMEN_TIER` env var equivalent for pesti)

#### Phase C: Tier-Specific Optimizations (Weeks 5-6)
- CPU baseline: Generic kernels with minimal overhead
- GpuFlashAttention: Flash attention + WGMMA kernel fusion
- GpuCudaOxide: Full cuda-oxide backend with shape-backed inline caches

**Timeline:** 3-5 weeks for MVP  
**Confidence:** Medium (requires performance profiling infrastructure)

---

## Phase 6: Model IR Abstraction (Early 2027) 🟠 Low-Medium Priority

### Goal
Single intermediate representation feeding multiple execution backends, inspired by pon's PON IR pattern.

### Current Gap
Weight loading → direct tensor ops → backend-specific kernels. Missing abstraction layer for kernel fusion and cross-backend benchmarking.

### Design Sketch

```rust
// New crate: llm-runner/crates/model-ir
pub enum ModelIR {
    Llama(LlamaGraph),
    Mistral(MistralGraph),
    Gemma(GemmaGraph),
}

pub struct LlamaGraph {
    layers: Vec<TransformerLayer>, // Abstracted layer representation (not backend-specific)
    config: LlamaConfig,           // Architecture-specific params
}

// Execution backends
pub enum ExecutionBackend {
    CpuCrate(CpuKernel),
    LlammaCppFfi(LlamaRunner),
    CandleCore(CandleBridge),
    GpuCudaOxide(CudaOxideBackend),
}

impl ModelIR {
    pub async fn infer(&self, backend: &mut ExecutionBackend, input: &Tensor) -> Result<Tensor> {
        // Build fused kernel graph based on available backends
        let graph = self.build_fused_graph(backend.available_backends())?;
        
        // Execute with auto-selection (CPU/GPU/remote priority)
        backend.execute(graph).await
    }
}

// Backend-agnostic inference loop
impl ModelIR {
    pub fn build_fused_graph(&self, backends: &[ExecutionBackend]) -> Result<FusedGraph> {
        match self {
            Self::Llama(ref llama) => Self::build_llama_fused_graph(llama, backends),
            Self::Mistral(ref mistral) => Self::build_mistral_fused_graph(mistral, backends),
            Self::Gemma(ref gemma) => Self::build_gemma_fused_graph(gemma, backends),
        }
    }
}

// Fused graph representation (backend-agnostic)
pub struct FusedGraph {
    nodes: Vec<OperationNode>,
    metadata: GraphMetadata,
}

pub enum OperationNode {
    MatMul(MatrixMultiplyOp),
    Attention(AttentionOp),
    LayerNorm(RmsNormOp),
    // ... other operations
}
```

### Implementation Phases

#### Phase A: Canonical Graph Representation (Weeks 1-4)
- Define `TransformerLayer`, `MatMul`, `Attention` ops in `model-ir` crate
- Support Llama/Mistral/Gemma architectures
- Add serialization/deserialization for graph persistence

#### Phase B: Backend Adapters (Weeks 5-8)
- Implement CPU kernel adapter
- Integrate llama.cpp FFI adapter (reuse existing code)
- Integrate candle-core adapter (reuse existing bridge)
- Implement cuda-oxide adapter (reuse existing backend)

#### Phase C: Kernel Fusion & Optimization (Weeks 9-12)
- Build fused operation graphs from canonical representation
- Enable cross-backend benchmarking harness
- Add tier-up threshold integration (Phase 5.2)

**Timeline:** 4-6 weeks for MVP  
**Confidence:** Low-Medium (significant refactoring required, but high long-term value)

---

## Integration Priority Matrix

| Pattern | Value | Effort | Current Gap | Recommended Timeline | Confidence |
|---------|-------|--------|--------------|---------------------|------------|
| **Differential conformance testing** | 🔴 Critical | 🟡 Medium (2-3 weeks) | GGUF v3 bug showed silent corruption | Phase 5.1 (next sprint) | High |
| **Tiered execution with tier-up thresholds** | 🟡 High | 🟠 High (3-5 weeks) | DeviceRouter exists; needs profiling loop | Phase 5.2 (Q4 2026) | Medium |
| **One IR for multiple backends** | 🟠 Medium-High | 🔴 Very High (4-6 weeks) | Weight loading is direct, not graph-based | Phase 6 (early 2027) | Low-Medium |

---

## Key Research Insights Applied to PESTI

### 1. Differential Testing is Non-Negotiable ✅
**Source:** Both lumen and pon use byte-exact comparison against references to catch silent corruption.  
**PESTI application:** GGUF v3 bug proved this pattern prevents regressions before they reach mainline.

### 2. Tiered Execution with Profile-Driven Optimization 🎯
**Source:** Lumen's tier-0→tier-1→tier-2 model mirrors pesti's need for CPU baseline → GPU specialized paths, but with runtime profiling to decide *when* to optimize.  
**PESTI application:** Leverage existing DeviceRouter infrastructure; add call-count tracking and tier-up threshold checks.

### 3. One IR Enables True Backend Abstraction 🎯
**Source:** Pon's single PON IR feeding both JIT and AoT compilers is architecturally cleaner than pesti's current direct weight loading pattern.  
**PESTI application:** Worth the refactoring effort for long-term flexibility; start with Phase A canonical graph representation.

### 4. Zero-Dependency Constraint is a Feature ✅
**Source:** Both projects run with std-only dependencies (no tokio, serde, libc).  
**PESTI application:** Maintain as design principle; pesti already follows this pattern.

---

## Next Session Action Items

1. **Fix remaining test assertions** (4 tests) - Convert `kv_pair_u32` → `kv_pair_u64` in model.rs line 905
2. **Review conformance testing MVP design** - Validate `pesti-conformance` crate structure
3. **Begin tier-up threshold profiling infrastructure** - Add call-count tracking to DeviceRouter

---

*End of roadmap update.*  
*Next session: Review PR for test fixes + begin conformance testing implementation*
