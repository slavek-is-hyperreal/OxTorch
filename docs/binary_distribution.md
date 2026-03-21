# OxTorch — Hardware-Native Binary Distribution

## Goal

Instead of shipping a single "generic" binary, each user receives a `.whl` file compiled for their specific CPU and GPU. This enables:
- Native instruction sets (`-C target-cpu=native`, AVX-512, AMX)
- Compile-time MSTS constants (tile size, ring buffer depth) tuned to the exact L2/L3 cache of the machine
- Vulkan shader variants generated for the specific GPU driver/vendor

---

## How It Works — Step by Step

```
[User Machine]                           [OxTorch Build Server]
        |                                              |
        | 1. curl -sSf https://get.oxtorch.io | bash  |
        |--------------------------------------------->|
        |                                              |
        | 2. Script collects hardware fingerprint:     |
        |    - cpuid  (micro-arch, AVX2/AVX512/AMX)   |
        |    - lspci  (GPU PCI ID -> Vulkan driver)    |
        |    - sysfs  (L2/L3 cache per core)           |
        |    - /proc/meminfo (available RAM)           |
        |                                              |
        | 3. HTTPS POST { fingerprint_sha256, ... } -->|
        |                                              |
        |                          4. cache hit? (S3)  |
        |                             yes -> skip 5    |
        |                             no  -> compile:  |
        |                          5. RUSTFLAGS="-C    |
        |                             target-cpu=..."   |
        |                             maturin build    |
        |                             --release        |
        |                             store .whl -> S3 |
        |                                              |
        | 6. { url: "https://cdn.../xyz.whl" }  <------|
        |                                              |
        | 7. pip install https://cdn.oxtorch.io/xyz.whl|
```

---

## pip Integration

Python supports this natively via **PEP 425 Platform Tags**. A standard `.whl` filename looks like:

```
oxtorch-3.7.0-cp312-cp312-linux_x86_64.whl
#       |      |     |     +-- platform tag
#       |      |     +-- ABI tag
#       |      +-- Python version
#       +-- package version
```

The build server generates wheels with a **custom platform tag** (e.g. `linux_x86_64_avx512`, `linux_x86_64_znver4`). Users install via:

```bash
pip install oxtorch \
  --extra-index-url https://pkg.oxtorch.io/wheels/$(oxtorch-detect)
```

Where `oxtorch-detect` is a small Rust CLI that outputs the machine's hardware fingerprint hash.

---

## Server-Side Build Variables

The hardware fingerprint maps to compile-time variables:

| Machine Parameter | Source | Rust Variable (`build.rs`) | Formula |
|---|---|---|---|
| L2 cache per core (KB) | `/sys/devices/system/cpu/cpu0/cache/index*/size` | `MSTS_TILE_BYTES` | `L2_KB * 1024 * 0.75` |
| L3 cache total (MB) | same, `index3` | `MSTS_RING_DEPTH` | `min(L3_MB / TILE_MB, 64)` |
| CPU micro-arch | `cpuid` leaf 0x1 via `raw-cpuid` crate | `RUSTFLAGS=-C target-cpu=...` | e.g. `znver4`, `alderlake` |
| SIMD flags | `cpuid` leaf 7 | `RUSTFLAGS=-C target-feature=...` | `+avx512f,+avx512vnni,...` |
| GPU PCI ID | `lspci -nn \| grep VGA` | `VNN_VULKAN_VENDOR` | `10de`=NVIDIA `1002`=AMD `8086`=Intel |
| GPU subgroup size | vendor lookup table | `VNN_SUBGROUP_SIZE` | NVIDIA=32 AMD=64 Intel=16 |
| VRAM (MB) | `vulkaninfo` or vendor sysfs | `VNN_VRAM_MB` | staging buffer sizing |
| Available RAM (GB) | `/proc/meminfo MemAvailable` | `VNN_RAM_BUDGET_GB` | MSTS prefetch depth |

---

## Example `build.rs` (skeleton)

```rust
fn main() {
    // Set by the build server at compile time
    let tile_bytes: usize = std::env::var("MSTS_TILE_BYTES")
        .unwrap_or_else(|_| "1048576".to_string())  // 1MB safe default
        .parse().unwrap();

    let ring_depth: usize = std::env::var("MSTS_RING_DEPTH")
        .unwrap_or_else(|_| "8".to_string())
        .parse().unwrap();

    println!("cargo:rustc-env=MSTS_TILE_BYTES={}", tile_bytes);
    println!("cargo:rustc-env=MSTS_RING_DEPTH={}", ring_depth);
}
```

In Rust code:
```rust
const TILE_BYTES: usize = env!("MSTS_TILE_BYTES").parse().unwrap_or(1_048_576);
```

---

## Server-Side Cache

To avoid recompiling for every user with the same hardware:
1. Hash fingerprint → S3 key.
2. If key exists → return pre-built URL (< 1 second).
3. If not → compile in Docker container (`cold build`, ~30–90 s), store on S3.

For the most common configurations (AMD Ryzen 7000 + RTX 3080, Intel 13th Gen + no GPU, etc.) wheels are **pre-built** and the response is instant.

---

## Security

- Fingerprint contains **hardware data only** — no personal info, no MAC addresses.
- HTTPS transport; wheel signed with OxTorch GPG key.
- Users can compile locally: `oxtorch-install --local` (requires Rust toolchain + maturin).

---

## TODO — Sprint 6

- [ ] `oxtorch-detect` CLI (Rust, outputs hardware fingerprint JSON)
- [ ] Build server (GitHub Actions self-hosted runner or Hetzner x86_64)
- [ ] S3 wheel cache + CDN (Cloudflare R2)
- [ ] `build.rs` reading `MSTS_TILE_BYTES` instead of hardcoded `1_048_576`
- [ ] Custom platform tags in `maturin` (`--target-platform`)
