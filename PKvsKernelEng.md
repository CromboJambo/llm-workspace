# Current State vs. vLLM Drop-in Replacement

**You:** I am building a rust-first stack and I am really not in a hurry, I just enjoy learning and solving problems.

**Hermes:** That's the best mindset for this. Let me map out the **interesting problems** ahead — not as a checklist, but as learning projects:

---

## The Fun Problems (in dependency order)

**You:** Oh yes! I want to apply my accounting pivot table concepts and query knowledge to work with my rust syntax knowledge and VR modding background to good use.

**Hermes:** This is a **perfect storm of mental models**. Let me map your backgrounds to WGMMA:

### Your Mental Models → GPU Kernel Concepts

| Your Background | Maps To | Why It Clicks |
|----------------|---------|---------------|
| **Pivot tables** (dimensions, measures, GROUP BY) | **Tensor contractions** | GEMM *is* a pivot: `M×K × K×N → M×N` where `K` is the aggregation dimension. Tiling = partitioning the pivot into pages. |
| **SQL query plans** (joins, aggregation, materialization) | **Kernel fusion + shared memory staging** | `cp.async` = materialized CTE. Double buffering = pipeline parallelism. Warp specialization = operator parallelism. |
| **Rust ownership/lifetimes** | **CUDA resource lifecycle** | `CudaContext` = `&'static mut Gpu`. `CudaStream` = `&mut CommandBuffer`. `DeviceBuffer<T>` = `Box<[T], GpuAllocator>`. RAII = no leaks. |
| **VR modding** (frame budget, draw calls, shader occupancy) | **Kernel launch + occupancy** | Launch = draw call (expensive). Occupancy = warps/SM (like vertex throughput). Shared memory = VRAM budget per block. TMA = async compute copy (like compute shader). |

---

## The WGMMA "Pivot Table" View

```
┌─────────────────────────────────────────────────────────────┐
│  GEMM: C[M, N] = Σₖ A[M, K] × B[K, N]                       │
│                                                             │
│  Pivot analogy:                                             │
│  - M = ROW dimension (customers)                            │
│  - N = COLUMN dimension (products)                          │
│  - K = AGGREGATION dimension (transactions)                 │
│  - A = fact table 1 (customer × transaction)                │
│  - B = fact table 2 (transaction × product)                 │
│  - C = pivot result (customer × product)                    │
└─────────────────────────────────────────────────────────────┘
```

**Tiling = Partitioning the pivot into chunks that fit in "fast memory" (shared memory)**

```
┌─────────────────────────────────────────────────────────────┐
│  WGMMA Tile: 64×64×16 (M×N×K) per warp group               │
│                                                             │
│  Shared memory "pivot cache":                               │
│  ┌─────────┬─────────┐                                       │
│  │  A tile │  B tile │  ← 64×16 + 16×64 = 2048 f16 = 4 KiB │
│  │ [64,16] │ [16,64] │                                       │
│  └─────────┴─────────┘                                       │
│                                                             │
│  Accumulator (registers): 64×64 f32 = 16 KiB per warp group │
│                                                             │
│  Double buffer: 2× A tiles + 2× B tiles = 16 KiB smem       │
└─────────────────────────────────────────────────────────────┘
```

---

## The VR Modder's Kernel Launch Checklist

| VR Concept | Kernel Equivalent | Target |
|------------|-------------------|--------|
| **Draw call overhead** | `cuLaunchKernel` | Batch work, persist kernels |
| **Vertex shader occupancy** | Warps/SM (max 64 on sm_120) | 4 warps × 8 blocks = 32 warps/SM |
| **VRAM bandwidth** | HBM → Shared memory via TMA | 1.5 TB/s peak, TMA gets ~90% |
| **Frame budget (16ms)** | Kernel latency budget | GEMM < 1ms for 4K tokens |
| **Double buffering** | `cp.async` stages 0/1 | Overlap load/compute |
| **Barrier sync** | `wgmma.commit_group` + `wgmma.wait_group` | Warp-group sync |

---

## Let's Build the Real WGMMA PTX

Starting point: your scalar stub in `llm-runner/src/kernel/ptx/gemm_wgmma.ptx`

**Target**: Replace the scalar K-loop with:

1. **TMA async copy** → load A/B tiles to shared memory
2. **WGMMA MMA** → tensor core multiply-accumulate
3. **Double buffering** → overlap copy + compute
4. **Warp specialization** → 2 warps load, 2 warps compute (optional v1)

Want to start with the **PTX structure** or the **Rust launch wrapper** first? The PTX is where the pivot-table logic lives; the Rust side is where your ownership model shines.
