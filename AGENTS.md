# PESTI Agent Rules

## Status

Phase 1 complete (CPU inference). Phase 1.5 complete (hybrid device routing). Phase 2 in progress (backend abstraction).

## Key Files

- `ROADMAP.md` — phased development roadmap with completion status
- `llmrunner.md` — LLM runner architecture notes
- `llm-runner/src/runner.rs` — DeviceRouter (Phase 1.5 wiring)
- `llm-runner/src/device.rs` — DeviceSelector + DeviceSelection
- `llm-runner/src/device_discovery.rs` — Local GPU enumeration
- `llm-runner/src/remote_discovery.rs` — Remote LM Studio discovery

## Architecture

```
pesti/
├── llm-runner/          Inference engine (CPU kernels operational, GPU stubbed)
├── gguf/                GGUF parser (all 29+ quantization types)
├── safetensors/         Weight storage + parsing
├── llm-plug-in/         Protocol + templates
├── cuda-oxide/          GPU host/device crates
└── rust-toolchain.toml  Nightly pinned
```

## Known Bug Patterns

### Format Version Mismatch in Test Data

When adding support for a new binary format version with a different wire layout
(e.g., GGUF V3: key lengths `u32` → `u64`), **always verify that test helpers
which generate synthetic test data use the new version's layout**, not the old
one. The parser will correctly implement the spec, but the test will silently
feed wrong-format data and fail with confusing "length exceeds max X" errors.

**Checklist when refactoring format versions:**
1. Diff the new parser's read calls against the test helper's write calls
2. Specifically check: integer widths (u32 vs u64), endianness, field ordering
3. If a `make_*_bytes()` helper exists for the version being changed, update it
4. Run tests with `RUST_BACKTRACE=1` — garbage-length errors are the telltale sign

### Hardware

| Device | VRAM | Free | Role |
|--------|------|------|------|
| GPU0 (RTX 4070) | 16GiB | ~1.6GiB | LM Studio |
| GPU1 (RTX 5060 Ti) | 16GiB | ~3.6GiB | LM Studio |
| Remote 3070 Ti | 8GiB | ~8GiB | Remote LM Studio |
