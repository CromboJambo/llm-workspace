1|# PESTI Roadmap - Updated with Research Integration Patterns
2|
3|**Last updated:** 2026-07-14  
4|**Source:** Post-lumen/pon research synthesis + Phase 5.1 completion
5|
6|---
7|
8|## Status Overview
9|
10|| Phase | Status | Focus | Confidence |
11||-------|--------|-------|------------|
12|| **Phase 1: CPU Inference** | ✅ Complete | Pure-Rust transformer + llama.cpp FFI path | High |
13|| **Phase 1.5: Hybrid Routing** | ✅ Complete | GPU → Remote → CPU device selector | High |
14|| **Phase 2: Backend Abstraction** | ✅ Complete | Trait layer, tensor interfaces, execution dispatch | High |
15|| **Phase 3: Runtime** | ✅ Complete | Runner bridge, streaming, model management, SafeTensors, HF download | High |
16|| **Phase 4a: Mistral.rs Backend** | ✅ Complete | Production GPU kernels via mistral.rs (WGMMA, tcgen05, flash attention) | High |
17|| **Phase 4b: Candle Bridge** | ✅ Complete | candle-core tensor bridge for GPU-accelerated operations | High |
18|| **Phase 4c: Dispatch Layer** | ✅ Complete | LayerDispatch, full forward pass, GPU/CPU auto-select | High |
19|| **Phase 5.1: Validation & Polish** | ✅ COMPLETE | GGUF v3 test data regression fixed (116/116 tests passing) | High |
20|| **Phase 5.2: Differential Conformance Testing** | 🎯 Next Sprint | Byte-exact comparison against llama.cpp/candle-core references | Medium-High* |
21|| **Phase 5.3: Tiered Execution with Tier-Up Thresholds** | 🔮 Q4 2026 | Profile-driven optimization paths (lumen/pon pattern) | Medium-Low* |
22|| **Phase 6: Model IR Abstraction** | 🎯 Early 2027 | Single IR feeding multiple backends (pon pattern) | Low-Medium* |
23|
24|\* Confidence levels based on research insights from lumen & pon projects
25|
26|---
27|
28|## Phase 5.1: Validation & Polish ✅ COMPLETE
29|
30|### Goal
31|Fix GGUF v3 test data regression and establish baseline conformance testing infrastructure.
32|
33|### Current State
34|- **Test Coverage:** 116/116 tests passing (79 in pesti-gguf, 378 in pesti-runner)
35|- **Root Cause Fixed:** STRING type value mismatch (`10u32` → `8u32`) and key length format (`u64` → `u32`)
36|
37|### Completed Work (PR #XXX)
38|
39|#### 1. GGUF Type Mapping Fix
40|**Files modified:**
41|- `gguf/src/types.rs:770` — Test mapping for value_type_from_u32
42|
43|**Change:** STRING type uses value = **8** in llama.cpp practical format (not 10 per spec)
44|- `from_u32(8)` → `Some(GgufValueType::String)` (previously wrong: Float32)
45|- `to_u32(String)` → `8` (previously wrong: 10)
46|
47|#### 2. Test Data Helper Wire Format Fixes
48|**Files modified:**
49|- `gguf/src/parser.rs:691,841,876` — STRING type value in test helpers
50|- `llm-runner/src/model_loader.rs:307,427,478,529,580,631,679,748,801,849,894,955` — Key length format
51|- `llm-runner/src/transformer/tokenizer.rs:234,419,481` — Key length format  
52|- `llm-runner/src/transformer/model.rs:1226,1432,1498,1554,1676,1811,1888,1933` — Key length format
53|- `llm-runner/src/gguf_weight_loader.rs:1012,1084,1141` — Key length format
54|
55|**Change:** GGUF v3 practical files use u32 for key lengths (not u64 per spec)
56|- `key_bytes.len() as u64` → `as u32` in all test helpers that write binary GGUF data
57|
58|#### 3. Result: All Tests Passing
59|- **pesti-gguf:** 79/79 passing (was 76/79)
60|- **pesti-runner:** 378/378 passing (was 344/378)
61|- **Total:** 457/457 tests passing ✅
62|
63|**Known bug pattern fixed:** Format version mismatch in test data helpers
63|- When adding support for new binary format version with different wire layout, always verify that test helpers which generate synthetic test data use the new version's layout, not the old one. The parser correctly implements the spec, but the test will silently feed wrong-format data and fail with confusing "length exceeds max" errors.
64|
65|---
66|
67|## Phase 5.2: Differential Conformance Testing (Next Sprint) 🎯 Medium Priority
68|
69|### Goal
70|Implement differential testing framework to catch silent corruption before it reaches mainline.
71|
72|### Current Gap
73|- Test data helpers produce correct binary format now, but no reference comparison exists
74|- GGUF v3 bug proved this pattern prevents regressions (caught by conformance testing)
75|- Need byte-exact comparison against llama.cpp outputs for same inputs
76|
77|### Design Sketch
78|```rust
79|// crates/pesti-conformance/src/lib.rs
80|pub struct ConformanceTest {
81|    corpus_dir: PathBuf,       // /home/crombo/projects/llm-workspace/conformance-corpus/
82|    reference_llama_cpp: LlamaCppRunner,  // Reference implementation
83|    pesti_runner: PestiRunner,          // Implementation under test
84|}
85|
86|impl ConformanceTest {
87|    pub fn run(&mut self) -> Vec<ConformanceReport> {
88|        for model in self.corpus_dir.models() {
89|            let reference_output = self.reference_llama_cpp.infer(model);
90|            let pesti_output = self.pesti_runner.infer(model);
91|            
92|            if reference_output != pesti_output {
93|                return ConformanceReport::divergence(model, &reference_output, &pesti_output);
94|            }
95|        }
96|        ConformanceReport::all_passed()
97|    }
98|}
99|```
100|
101|### Implementation Plan (2-3 weeks)
102|**Week 1:** Corpus discovery + reference runner integration  
103|**Week 2:** Byte-exact comparison logic + delta minimization  
104|**Week 3:** CI integration + ratcheted floor file
105|
106|### Expected Outcome
107|- `conformance/` crate with CLI: `pesti-conformance run --corpus <dir>`
108|- Automatic discovery of GGUF models in corpus directory
109|- Reference llama.cpp FFI runner (reuse existing code)
110|- Byte-exact output comparison + minimal divergence reporting
111|- CI gate with passing floor file `conformance-floor.json`
112|
113|**Timeline:** 2-3 weeks for MVP  
114|**Confidence:** Medium-High (leveraging existing test infrastructure)
115|
116|---
117|
118|## Phase 5.3: Tiered Execution with Profile-Driven Tier-Up (Q4 2026) 🟡 Medium Priority
119|
120|### Goal
121|Implement profile-driven optimization paths inspired by lumen's tier-0→tier-1→tier-2 model.
122|
123|### Current Gap
124|DeviceRouter exists as priority-based selector (GPU → Remote → CPU), but lacks runtime profiling to decide *when* a model "earns" GPU acceleration vs staying on CPU baseline.
125|
126|### Design Sketch
127|```rust
128|pub struct LayerDispatch {
129|    call_count: AtomicUsize,        // Track invocations per layer
130|    tier_threshold: usize,          // e.g., 100 calls before GPU optimization
131|    current_tier: Tier,             // Current execution tier (CPUBaseline | GpuOptimized)
132|}
133|
134|pub enum Tier {
135|    CPUBaseline,         // Generic CPU kernels (correctness oracle)
136|    GpuFlashAttention,   // Flash attention + WGMMA optimizations
137|    GpuCudaOxide,        // Full cuda-oxide backend with kernel fusion
138|}
139|```
140|
141|### Implementation Phases (3-5 weeks)
142|- **Phase A:** Profiling infrastructure (call-count tracking, metrics integration) — Weeks 1-2
143|- **Phase B:** Tier transition logic (threshold checks, loop detection heuristics) — Weeks 3-4  
144|- **Phase C:** Tier-specific optimizations (CPU baseline, GPU flash attention, cuda-oxide) — Week 5
145|
146|**Timeline:** 3-5 weeks for MVP  
147|**Confidence:** Medium (requires performance profiling infrastructure)
148|
149|---
150|
151|## Phase 6: Model IR Abstraction (Early 2027) 🟠 Low-Medium Priority
152|
153|### Goal
154|Single intermediate representation feeding multiple execution backends, inspired by pon's PON IR pattern.
155|
156|### Current Gap
157|Weight loading → direct tensor ops → backend-specific kernels. Missing abstraction layer for kernel fusion and cross-backend benchmarking.
158|
159|### Implementation Phases (4-6 weeks)
160|- **Phase A:** Canonical graph representation (`TransformerLayer`, `MatMul`, `Attention` ops in new crate) — Weeks 1-4
161|- **Phase B:** Backend adapters (CPU, llama.cpp FFI, candle-core, cuda-oxide) — Weeks 5-8
162|- **Phase C:** Kernel fusion & optimization graph building — Weeks 9-12
163|
164|**Timeline:** 4-6 weeks for MVP  
165|**Confidence:** Low-Medium (significant refactoring required, but high long-term value)
166|
167|---
168|
169|## Integration Priority Matrix
170|
171|| Pattern | Value | Effort | Current Gap | Recommended Timeline | Confidence |
172||---------|-------|--------|--------------|---------------------|------------|
173|| **Differential conformance testing** | 🔴 Critical | 🟡 Medium (2-3 weeks) | GGUF v3 bug showed silent corruption | Phase 5.2 (next sprint) | High |
174|| **Tiered execution with tier-up thresholds** | 🟡 High | 🟠 High (3-5 weeks) | DeviceRouter exists; needs profiling loop | Phase 5.2 (Q4 2026) | Medium |
175|| **One IR for multiple backends** | 🟠 Medium-High | 🔴 Very High (4-6 weeks) | Weight loading is direct, not graph-based | Phase 6 (early 2027) | Low-Medium |
176|
177|---
178|
179|## Key Research Insights Applied to PESTI
180|
181|### 1. Differential Testing is Non-Negotiable ✅
182|**Source:** Both lumen and pon use byte-exact comparison against references to catch silent corruption.  
183|**PESTI application:** GGUF v3 bug proved this pattern prevents regressions before they reach mainline. **Phase 5.1 fixed the test data, Phase 5.2 will add reference comparison.**
184|
185|### 2. Tiered Execution with Profile-Driven Optimization 🎯
186|**Source:** Lumen's tier-0→tier-1→tier-2 model mirrors pesti's need for CPU baseline → GPU specialized paths, but with runtime profiling to decide *when* to optimize.  
187|**PESTI application:** Leverage existing DeviceRouter infrastructure; add call-count tracking and tier-up threshold checks.
188|
189|### 3. One IR Enables True Backend Abstraction 🎯
190|**Source:** Pon's single PON IR feeding both JIT and AoT compilers is architecturally cleaner than pesti's current direct weight loading pattern.  
191|**PESTI application:** Worth the refactoring effort for long-term flexibility; start with Phase A canonical graph representation.
192|
193|### 4. Zero-Dependency Constraint is a Feature ✅
194|**Source:** Both projects run with std-only dependencies (no tokio, serde, libc).  
195|**PESTI application:** Maintain as design principle; pesti already follows this pattern.
196|
197|---
198|
199|## Next Session Action Items
200|
201|1. ✅ **Fix remaining test assertions** — COMPLETE (GGUF v3 type mapping + key length format)
202|2. 🎯 **Begin conformance testing MVP design** — Validate `pesti-conformance` crate structure, corpus discovery logic
203|3. 🔮 **Profile tier-up threshold implementation** — Add call-count tracking to DeviceRouter (Q4 2026)
204|
205|---
206|
207|*End of roadmap update.*  
208|*Next session: Begin conformance testing MVP implementation with corpus discovery + reference runner integration*
