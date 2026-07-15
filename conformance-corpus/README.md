# Conformance Corpus Directory

This directory holds GGUF models used for differential conformance testing against reference implementations.

## Purpose

- **Differential Testing**: Compare pesti outputs against llama.cpp/candle-core references to catch silent corruption
- **CI Gate**: Floor file ensures minimum passing count before merging
- **Regression Detection**: Byte-exact comparison reveals subtle numerical drift

## Model Selection Guidelines

For early-stage conformance (Phase 5.2), focus on:

1. **Small models** (< 2B params) - fast test cycles
2. **Q4_0 / Q8_0 quantizations** - covers most common use cases  
3. **Llama architecture** - primary target for pesti
4. **Minimal tokenizers** - reduces noise from tokenizer mismatches

Example models:
- `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (tiny, fast)
- `Llama-3.1-8B-Instruct-Q4_0.gguf` (popular, well-tested)
- `Mistral-7B-v0.3-Q8_0.gguf` (architectural diversity)

## Adding New Models

```bash
# Download to this directory
curl -L https://huggingface.co/<repo>/resolve/main/<model>.gguf -o conformance-corpus/<model>.gguf

# Verify file integrity
sha256sum conformance-corpus/<model>.gguf

# Run test against reference
cargo run --package pesti-conformance -- run \
    --corpus ./conformance-corpus/ \
    --reference-llama-cpp /usr/bin/llama-cli \
    --floor-file ./conformance-floor.json
```

## Floor File Format

```bash
# conformance-floor.json contains single integer: minimum passing count
25
```

CI checks that `passed.len() >= floor_pass_count`.

## Running Tests

### Without reference (baseline)
```bash
cargo run --package pesti-conformance -- run \
    --corpus ./conformance-corpus/ \
    --floor-pass-count 0
```

### With llama.cpp reference
```bash
cargo run --package pesti-conformance -- run \
    --corpus ./conformance-corpus/ \
    --reference-llama-cpp /home/crombo/.local/bin/llama-cli \
    --floor-pass-count 10
```

## Known Limitations (Early Phase)

1. **Determinism**: llama.cpp CLI now uses deterministic sampling (`--temp 0.0`); output hashes should be stable across runs
2. **Reference availability**: Requires external `llama-cli` binary built from source with CUDA support for GPU acceleration
3. **Model compatibility**: Not all GGUF files load in both pesti and llama.cpp (quantization quirks)

**Note:** For true byte-exact comparison, ensure llama.cpp was built with consistent floating-point settings (`-DGGML_CUDA=1` or similar).

### Mitigation: Deterministic Sampling

Phase 5.2 now uses `--temp 0.0` for deterministic argmax sampling instead of stochastic sampling (`--temp 0.8`). This means:
- Same input → same output (across runs)
- Hash comparison is meaningful for regression detection
- Still may differ between CPU/GPU due to floating-point precision differences

**For GPU-first workflows:** Consider building llama.cpp with `-DGGML_CUDA=1` and running conformance on the same device backend.

## Next Steps (Phase 5.3+)

- Add deterministic sampling mode (`--temp 0.0`) for exact byte-matching
- Integrate candle-core GPU reference path
- Add layer-by-layer divergence reporting via `delta_minimize()` API
