# OxTorch Binary Distribution Strategy

OxTorch focuses on providing high performance for specific hardware configurations, which requires a specialized approach to binary distribution.

---

## 1. Hardware Compilation (Native)

Unlike generic PyTorch packages, OxTorch is often compiled with `-C target-cpu=native`. This allows the Rust compiler to:
*   Utilize all specifically available SIMD instructions (e.g., F16C, AVX2, AVX-512).
*   Optimize the pipeline for the specific cache structure (L1/L2/L3) of the processor.
*   "Burn-in" MSTS thresholds directly into the binary based on local benchmark results (`build.rs`).

---

## 2. Pypi / Pip Integration

OxTorch is distributed via `.whl` files. We maintain a "Support Matrix" of pre-compiled binaries for the most popular CPU architectures (e.g., `x86_64-v3`, `arm64`).

*   **Auto-detection**: The installation script (`oxtorch-install`) detects the current CPU capabilities and pulls the optimal WHEEL from the repository.
*   **Signing**: All official binaries are signed with the OxTorch GPG key.

---

## 3. Local Compilation (`--local`)

For users with specialized hardware or non-standard server environments:
```bash
pip install oxtorch --install-option="--local"
```
This command triggers a local compilation on the target machine, ensuring 100% architectural fit and maximum performance (recommended for HPC clusters).

---

## 4. Hardware Fingerprinting

During the build process, the system collects hardware data (CPU model, SIMD support, NVMe speed). This is used solely for selecting the correct build and does **not** include personal information.

---

## 5. Offline Distribution

OxTorch provides a `bundle` package containing all necessary dependencies for deployment in air-gapped environments (Secure Enclaves / Mobile On-device).
