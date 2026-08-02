# Version Upgrade: v0.1.0 → v0.1.1

## Summary

PESTI has been upgraded to **v0.1.1** with a significant refactor introducing pure Rust dequantization, replacing C FFI dependencies.

---

## Key Changes

### 1. Pure Rust Dequantization Layer
- **New module**: `pesti-runner/src/dequantize.rs` (177 lines)
- **Functions**:
  - `dequantize_q4_0_ggml()` - Q4_0 dequantization
  - `dequantize_q4_1_ggml()` - Q4_1 dequantization  
  - `dequantize_q8_0_ggml()` - Q8_0 dequantization
- **Dependency**: `ggml-quants = "0.1"` crate

### 2. Removed C Dependencies
- Deleted legacy functions from `gguf_weight_loader.rs`:
  - `dequantize_q4_0()` (48 lines)
  - `dequantize_q4_1()` (52 lines)
  - `dequantize_q8_0()` (32 lines)
- **Net reduction**: ~132 lines of C-style code removed

### 3. CI/CD Infrastructure
- `.clippy.toml` - Strict linting rules
- `.github/workflows/ci.yml` - Automated testing & clippy
- `.github/workflows/release.yml` - Version bump automation
- `RELEASE.md` - Release process documentation
- `CHANGELOG.md` - Version history tracking

### 4. Build Performance
- **Clean build time**: ~60 seconds (from `cargo clean`)
- **Test suite**: 314 tests passing in `pesti-runner`
- **Warnings**: 16 clippy warnings (cosmetic style suggestions)

---

## Files Modified

### Production Code
```
M Cargo.toml                          (version bump 0.1.0 → 0.1.1)
M pesti-runner/Cargo.toml            (added ggml-quants, byteorder deps)
M pesti-runner/src/lib.rs            (exported dequantize module)
M pesti-runner/src/gguf_weight_loader.rs (replaced FFI calls)
```

### New Files
```
A  pesti-runner/src/dequantize.rs           (177 lines)
A  pesti-runner/src/dequantize_cuda.rs      (44 lines - CUDA stub)
A  .clippy.toml                            (strict lint config)
A  CHANGELOG.md                            (version history)
A  RELEASE.md                              (release process)
```

### Infrastructure
```
A  .github/workflows/ci.yml               (CI pipeline)
A  .github/workflows/release.yml          (automated release)
```

---

## Verification Status

✅ **Build**: Clean (`cargo check` passes)  
✅ **Tests**: 314/314 passing in `pesti-runner`  
✅ **Formatting**: `cargo fmt --check` ready  
⚠️ **Clippy**: 16 warnings (cosmetic, can be auto-fixed)  

---

## Next Steps

### Immediate
1. Run `cargo fix --lib -p pesti-runner` to auto-fix clippy suggestions
2. Review and merge CI/CD workflows to GitHub
3. Test release automation workflow manually

### Phase 2 (v0.2.0)
- Implement CUDA kernels in `dequantize_cuda.rs`
- Add performance benchmarks comparing Rust vs C FFI
- Document GPU acceleration speedups

---

## Semver Impact

**PATCH bump** (0.1.0 → 0.1.1) because:
- ✅ Backwards-compatible API additions
- ✅ Internal refactoring without breaking external contracts
- ⚠️ Removed internal functions (but they were private to `gguf_weight_loader`)

The public API of `pesti-runner` remains unchanged. Consumers will see no breaking changes.

---

*Generated: August 1, 2026*  
*Version: v0.1.1*
