# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.0] - 2026-03-09 "Valkyrie"
### Added
- **Tri-Precision Engine**: Native support for `DataType::F32`, `DataType::F16`, and `DataType::BF16`.
- **Statistical Safety Net**: Multi-run benchmarking (`--runs N`) with Median, Mean, and StdDev metrics.
- **Session Duration Tracking**: Recorded `total_duration_seconds` for hardware-level thermal analysis.
- **Comprehensive Documentation**: Complete overhaul of `README.md` and `docs/` with accurate source-line references.

### Fixed
- **PyTorch F16/BF16 Parity**: Adjusted tolerances for low-precision backends.
- **Thermal Noise Suppression**: Median results used in regression tracking to filter system interference.

## [3.1.2] - 2026-03-05
### Changed
- **Optimized CPU Fallback**: Improved conversion speed for F16 compute on legacy hardware.
- **Benchmark Iteration Sharding**: Reduced iterations for slow F16 MatMuls to decrease audit duration.

## [2.9.0] - 2026-03-04
### Added
- **Statistical Guard**: Integrated Coefficient of Variation (CV%) and P95 percentile tracking in benchmarks to filter system noise.
- **API Reference**: Detailed documentation of all Rust/Python bindings with source line references.
- **Hardware-Invariant Monitoring**: Switched to Ratio (VNN/PT) for regression detection.
