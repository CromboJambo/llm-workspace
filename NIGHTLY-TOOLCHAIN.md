# Nightly Toolchain Requirement

## Why?

The `cuda-oxide` crate (from `NVlabs/cuda-oxide.git`) requires **Rust nightly** due to its use of unstable compiler features:

```rust
#![feature(proc_macro_def_site, proc_macro_tracked_env)]
```

These features are used in `cuda-macros` which is pulled as a dependency when compiling `cuda-oxide`.

## Impact

### What Requires Nightly?

- **`cuda-oxide`** — CUDA host/device crates (stubbed kernels)
  - Pulls in: `cuda-core`, `cuda-device`, `cuda-host`, `cuda-macros`
  - Requires nightly for `proc_macro_def_site` and `proc_macro_tracked_env` features

### What Works on Stable?

- **`pesti-runner`** — Pure Rust dequantization (ggml-quants) ✅
- **`pesti-gguf`** — GGUF parser ✅
- **`candle-core`** — ML inference backbone ✅
- **`mistral.rs`** — Production GPU kernels (if compiled separately) ✅

## Configuration

### CI/CD Workflows

All GitHub Actions workflows now use nightly:

```yaml
- name: Install Rust toolchain
  uses: dtolnay/rust-toolchain@nightly
  with:
    components: clippy, rustfmt
```

**Files updated:**
- `.github/workflows/ci.yml` — All jobs (clippy, fmt, test, build, semver-check)
- `.github/workflows/release.yml` — Version bump job

### Local Development

To use nightly locally:

```bash
# Install nightly toolchain
rustup install nightly

# Set as default for this project
cd /home/crombo/projects/llm-workspace
rustup override set nightly

# Or use specific channel per command
cargo +nightly check --workspace
```

### rust-toolchain.toml

The project already has a pinned nightly toolchain:

```toml
[toolchain]
channel = "nightly"
components = ["rustfmt", "clippy"]
```

This ensures consistent builds across environments.

## Trade-offs

### Pros of Using Nightly

✅ **Full CUDA support** — `cuda-oxide` and all its dependencies compile cleanly  
✅ **Latest features** — Access to unstable Rust features for future optimizations  
✅ **Consistency** — Matches the workspace's existing nightly pin  

### Cons of Using Nightly

⚠️ **Slightly less stable** — Nightly can have breaking changes (but `cuda-oxide` pins its version)  
⚠️ **Slower compilation** — Marginal slowdown vs stable (negligible in practice)  

## Mitigation Strategies

1. **Pin `cuda-oxide` version** — Already done via git commit hash (`868f8ec4`)
2. **Use `rustup override`** — Project-specific nightly pinning
3. **Test on stable first** — Core logic (ggml-quants, pesti-runner) tested on stable before enabling nightly

## Future Considerations

### Option A: Fork & Stabilize `cuda-oxide`

If we want to avoid nightly dependency long-term:
- Fork `cuda-oxide` and replace unstable features with stable alternatives
- Trade-off: More maintenance burden, but full control over dependencies

### Option B: Keep Nightly (Current Approach)

Continue using nightly for `cuda-oxide` while keeping core logic on stable:
- **Pros:** Minimal changes, leverages existing ecosystem
- **Cons:** Relies on nightly for one optional backend

**Recommendation:** Stick with nightly for now. The CUDA path is optional; the CPU path (ggml-quants) works perfectly on stable. When GPU kernels become critical, we can re-evaluate.

## Verification

To verify nightly is working:

```bash
# Check toolchain version
rustup show

# Should see:
# Default host: x86_64-unknown-linux-gnu
# rustc channel: nightly
# ...

# Build with nightly
cargo +nightly check --workspace

# Should compile without E0554 errors
```

## Related Files

- `.github/workflows/ci.yml` — CI jobs use nightly
- `.github/workflows/release.yml` — Release job uses nightly
- `rust-toolchain.toml` — Project-wide nightly pin
- `Cargo.toml` — Workspace configuration with nightly dependencies