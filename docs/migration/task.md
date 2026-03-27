# Task List: CPU Backend Granular Migration (HPC)

## Phase 0: Infrastructure & Skeleton (W TOKU)
- `[x]` **0.1** Backup: `mv src/cpu src/cpu_old`
- `[x]` **0.2** Update `src/lib.rs`: `pub mod cpu; pub mod cpu_old;`
- `[x]` **0.3** Hierarchy Skeleton:
  - `[x]` `src/cpu/mod.rs`
  - `[x]` `src/cpu/ops/mod.rs`
  - `[x]` `src/cpu/ops/binary/mod.rs`
- `[x]` **0.4** Initial Report: `docs/migration/kernel_report.md`
- `[/]` **0.5** Verification: `cargo build` (Verify that `cpu_old` shims work)

## Phase 1: Binary Operations Atomization

### Op: `add` (Wzór)
- `[ ]` **1.add.bf16**:
  - `[x]` `add_bf16_generic.rs`
  - `[/]` `add_bf16_avx.rs` (W TOKU)
  - `[ ]` `add_bf16_neon.rs`
  - `[ ]` `bf16/mod.rs` (Static Dispatch)
- `[ ]` **1.add.f16**:
- `[ ]` **1.add.f32**:
- `[ ]` **1.add.i8**:

### Op: `sub`
- `[ ]` **1.sub.bf16**:
- `[ ]` **1.sub.f16**:
- `[ ]` **1.sub.f32**:
- `[ ]` **1.sub.i8**:

### Op: `mul`
- `[ ]` **1.mul.bf16**:
- `[ ]` **1.mul.f16**:
- `[ ]` **1.mul.f32**:
- `[ ]` **1.mul.i8**:

### Op: `div`
- `[ ]` **1.div.bf16**:
- `[ ]` **1.div.f16**:
- `[ ]` **1.div.f32**:
- `[ ]` **1.div.i8**:

## Phase 2: Verification & Cleanup
- `[ ]` **2.1** Remove `src/cpu_old` after full parity passed
- `[ ]` **2.2** Benchmarking
- `[ ]` **2.3** Finalize `kernel_report.md`
