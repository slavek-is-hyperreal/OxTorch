//! benches/kernels.rs — per-architecture microbenchmarks for the CPU kernels.
//!
//! # Why this file exists
//! These benchmarks call the architecture-specific kernels **directly**
//! (`sub_f32_avx1`, `sub_f32_sse2`, `sub_f32_scalar`, …), bypassing the Tier II
//! dispatch ladder and the Tier III rayon gate. That is the only way to get an
//! honest per-variant number: measured from Python — or through the Tier II
//! entry point — dispatch always picks the widest variant the CPU supports, so
//! the SSE2 and scalar paths are simply never observed.
//!
//! **The `// BENCH:` headers in kernel source files must be sourced from here.**
//! A number obtained any other way is measuring the dispatcher, not the kernel.
//!
//! # Running
//! ```text
//! cargo bench --bench kernels                    # everything
//! cargo bench --bench kernels -- sub_f32         # one op
//! cargo bench --bench kernels -- --test          # compile + one iteration (CI)
//! ```
//!
//! # Adding an op (later waves)
//! 1. Write `fn bench_<op>(c: &mut Criterion)` below, modelled on `bench_sub_f32`.
//! 2. Add it to the `criterion_group!` list at the bottom.
//! One `criterion_group` function per op keeps merge conflicts to a single line.
//!
//! # Reading the numbers
//! Sizes are chosen to sit in distinct levels of the memory hierarchy; on the
//! reference box (i5-3450, 4x 32 KiB L1d, 4x 256 KiB L2, 6 MiB L3) an elementwise
//! binary op is bandwidth-bound above L2, so above ~64 Ki elements all variants
//! converge and the interesting differences live in the L1/L2 sizes.
//! Non-temporal stores (`vmovntps`, used by the AVX1/AVX2 kernels here) only pay
//! off once the working set exceeds L3 — expect them to *lose* at small sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Element counts to sweep. Deliberately includes a non-multiple-of-64 size so a
/// kernel that is fast only on perfectly aligned lengths cannot hide.
const SIZES: &[usize] = &[
    4_096,     // 16 KiB/buffer  — fits L1d
    65_536,    // 256 KiB/buffer — fits L2
    1_048_576, // 4 MiB/buffer   — exceeds L3 for 3 buffers
    1_048_573, // same, prime-ish length: exercises the scalar tail
];

/// 64-byte-aligned buffer.
///
/// **This is not a micro-optimisation, it is a correctness requirement.** The
/// AVX1/AVX2 kernels in this repo store with `vmovntps` / `_mm256_stream_ps`,
/// whose memory operand must be 32-byte aligned; an unaligned destination raises
/// #GP and the process dies with SIGSEGV. A plain `vec![0f32; n]` is only
/// 8/16-byte aligned in practice, so benching those kernels on one crashes.
/// 64 bytes matches the alignment `TensorPool` guarantees for real tensors.
pub struct Aligned64 {
    buf: Vec<f32>,
    off: usize,
    len: usize,
}

impl Aligned64 {
    fn new(len: usize) -> Self {
        // 16 extra f32 = 64 bytes of slack to slide the start into alignment.
        let buf = vec![0f32; len + 16];
        let off = (64 - (buf.as_ptr() as usize % 64)) % 64 / std::mem::size_of::<f32>();
        let me = Self { buf, off, len };
        debug_assert_eq!(me.as_slice().as_ptr() as usize % 64, 0);
        me
    }

    fn filled(len: usize, seed: u32) -> Self {
        let mut me = Self::new(len);
        let data = make_f32(len, seed);
        me.as_mut_slice().copy_from_slice(&data);
        me
    }

    fn as_slice(&self) -> &[f32] {
        &self.buf[self.off..self.off + self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.buf[self.off..self.off + self.len]
    }
}

fn make_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 16.0
        })
        .collect()
}

fn make_bf16(n: usize, seed: u32) -> Vec<half::bf16> {
    make_f32(n, seed).into_iter().map(half::bf16::from_f32).collect()
}

/// Register one binary f32 variant, if the host actually supports it.
///
/// `f` is invoked inside the timing loop with no dispatch in between.
fn bench_binary_f32_variant(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    supported: bool,
    n: usize,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    f: impl Fn(&[f32], &[f32], &mut [f32]),
) {
    if !supported {
        return;
    }
    group.bench_with_input(BenchmarkId::new(label, n), &n, |bencher, _| {
        bencher.iter(|| {
            f(black_box(a), black_box(b), black_box(&mut *out));
            black_box(out.len())
        })
    });
}

// ---------------------------------------------------------------------------
// sub / f32
// ---------------------------------------------------------------------------
fn bench_sub_f32(c: &mut Criterion) {
    use vulkannn_rusted::cpu::ops::binary::sub::fp32;

    let mut group = c.benchmark_group("sub_f32");
    for &n in SIZES {
        let a = Aligned64::filled(n, 0x9E37_79B9);
        let b = Aligned64::filled(n, 0x85EB_CA6B);
        let mut out = Aligned64::new(n);
        let (a, b) = (a.as_slice(), b.as_slice());
        group.throughput(Throughput::Bytes((n * 4 * 3) as u64));

        bench_binary_f32_variant(
            &mut group, "scalar", true, n, a, b, out.as_mut_slice(),
            |x, y, r| fp32::sub_f32_scalar::sub_f32_scalar(x, y, r),
        );

        #[cfg(target_arch = "x86_64")]
        {
            bench_binary_f32_variant(
                &mut group, "sse2", is_x86_feature_detected!("sse2"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::sub_f32_sse2::sub_f32_sse2(x, y, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx1", is_x86_feature_detected!("avx"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::sub_f32_avx1::sub_f32_avx1(x, y, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx2", is_x86_feature_detected!("avx2"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::sub_f32_avx2::sub_f32_avx2(x, y, r) },
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            bench_binary_f32_variant(
                &mut group, "neon", std::arch::is_aarch64_feature_detected!("neon"),
                n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::sub_f32_neon::sub_f32_neon(x, y, r) },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// add / f32
// ---------------------------------------------------------------------------
fn bench_add_f32(c: &mut Criterion) {
    use vulkannn_rusted::cpu::ops::binary::add::fp32;

    let mut group = c.benchmark_group("add_f32");
    for &n in SIZES {
        let a = Aligned64::filled(n, 0x9E37_79B9);
        let b = Aligned64::filled(n, 0x85EB_CA6B);
        let mut out = Aligned64::new(n);
        let (a, b) = (a.as_slice(), b.as_slice());
        group.throughput(Throughput::Bytes((n * 4 * 3) as u64));

        bench_binary_f32_variant(
            &mut group, "scalar", true, n, a, b, out.as_mut_slice(),
            |x, y, r| fp32::add_f32_scalar::add(x, y, r),
        );

        #[cfg(target_arch = "x86_64")]
        {
            bench_binary_f32_variant(
                &mut group, "avx1", is_x86_feature_detected!("avx"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::add_f32_avx1::add(x, y, r) },
            );
            // NOTE: add has no SSE2 kernel yet — a gap a later wave should fill,
            // since it is the only path on pre-Sandy-Bridge hardware.
            bench_binary_f32_variant(
                &mut group, "avx2", is_x86_feature_detected!("avx2"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::add_f32_avx2::add(x, y, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx512", is_x86_feature_detected!("avx512f"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::add_f32_avx512::add(x, y, r) },
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            bench_binary_f32_variant(
                &mut group, "neon", std::arch::is_aarch64_feature_detected!("neon"),
                n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::add_f32_neon::add(x, y, r) },
            );
        }
    }
    group.finish();
}

// mul (memory-bound; plain cached stores — expect SIMD >= scalar at all N)
fn bench_mul_f32(c: &mut Criterion) {
    use vulkannn_rusted::cpu::ops::binary::mul::fp32;

    let mut group = c.benchmark_group("mul_f32");
    for &n in SIZES {
        let a = Aligned64::filled(n, 0x9E37_79B9);
        let b = Aligned64::filled(n, 0x85EB_CA6B);
        let mut out = Aligned64::new(n);
        let (a, b) = (a.as_slice(), b.as_slice());
        group.throughput(Throughput::Bytes((n * 4 * 3) as u64));

        bench_binary_f32_variant(
            &mut group, "scalar", true, n, a, b, out.as_mut_slice(),
            |x, y, r| fp32::mul_f32_scalar::mul(x, y, r),
        );

        #[cfg(target_arch = "x86_64")]
        {
            bench_binary_f32_variant(
                &mut group, "sse2", is_x86_feature_detected!("sse2"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::mul_f32_sse2::mul_f32_sse2(x, y, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx1", is_x86_feature_detected!("avx"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::mul_f32_avx1::mul_f32_avx1(x, y, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx2", is_x86_feature_detected!("avx2"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::mul_f32_avx2::mul_f32_avx2(x, y, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx512", is_x86_feature_detected!("avx512f"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::mul_f32_avx512::mul_f32_avx512(x, y, r) },
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            bench_binary_f32_variant(
                &mut group, "neon", std::arch::is_aarch64_feature_detected!("neon"),
                n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::mul_f32_neon::mul_f32_neon(x, y, r) },
            );
        }
    }
    group.finish();
}

// div (higher-latency divide — SIMD can win more than mul even when bandwidth-bound)
fn bench_div_f32(c: &mut Criterion) {
    use vulkannn_rusted::cpu::ops::binary::div::fp32;

    let mut group = c.benchmark_group("div_f32");
    for &n in SIZES {
        let a = Aligned64::filled(n, 0x9E37_79B9);
        let b = Aligned64::filled(n, 0x85EB_CA6B);
        let mut out = Aligned64::new(n);
        let (a, b) = (a.as_slice(), b.as_slice());
        group.throughput(Throughput::Bytes((n * 4 * 3) as u64));

        bench_binary_f32_variant(
            &mut group, "scalar", true, n, a, b, out.as_mut_slice(),
            |x, y, r| fp32::div_f32_scalar::div(x, y, r),
        );

        #[cfg(target_arch = "x86_64")]
        {
            bench_binary_f32_variant(
                &mut group, "sse2", is_x86_feature_detected!("sse2"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::div_f32_sse2::div_f32_sse2(x, y, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx1", is_x86_feature_detected!("avx"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::div_f32_avx1::div_f32_avx1(x, y, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx2", is_x86_feature_detected!("avx2"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::div_f32_avx2::div_f32_avx2(x, y, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx512", is_x86_feature_detected!("avx512f"), n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::div_f32_avx512::div_f32_avx512(x, y, r) },
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            bench_binary_f32_variant(
                &mut group, "neon", std::arch::is_aarch64_feature_detected!("neon"),
                n, a, b, out.as_mut_slice(),
                |x, y, r| unsafe { fp32::div_f32_neon::div_f32_neon(x, y, r) },
            );
        }
    }
    group.finish();
}

// atan2 (compute-bound: polynomial eval — expect a clean SIMD win at every N)
fn bench_atan2_f32(c: &mut Criterion) {
    use vulkannn_rusted::cpu::ops::binary::atan2::fp32;

    let mut group = c.benchmark_group("atan2_f32");
    for &n in SIZES {
        let a = Aligned64::filled(n, 0x9E37_79B9);
        let b = Aligned64::filled(n, 0x85EB_CA6B);
        let mut out = Aligned64::new(n);
        let (a, b) = (a.as_slice(), b.as_slice());
        group.throughput(Throughput::Bytes((n * 4 * 3) as u64));

        bench_binary_f32_variant(
            &mut group, "scalar", true, n, a, b, out.as_mut_slice(),
            |y, x, r| fp32::atan2_f32_scalar::atan2(y, x, r),
        );

        #[cfg(target_arch = "x86_64")]
        {
            bench_binary_f32_variant(
                &mut group, "avx1", is_x86_feature_detected!("avx"), n, a, b, out.as_mut_slice(),
                |y, x, r| unsafe { fp32::atan2_f32_avx1::atan2(y, x, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx2", is_x86_feature_detected!("avx2"), n, a, b, out.as_mut_slice(),
                |y, x, r| unsafe { fp32::atan2_f32_avx2::atan2(y, x, r) },
            );
            bench_binary_f32_variant(
                &mut group, "avx512", is_x86_feature_detected!("avx512f"), n, a, b, out.as_mut_slice(),
                |y, x, r| unsafe { fp32::atan2_f32_avx512::atan2(y, x, r) },
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            bench_binary_f32_variant(
                &mut group, "neon", std::arch::is_aarch64_feature_detected!("neon"),
                n, a, b, out.as_mut_slice(),
                |y, x, r| unsafe { fp32::atan2_f32_neon::atan2(y, x, r) },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// sub / bf16
// ---------------------------------------------------------------------------
fn bench_sub_bf16(c: &mut Criterion) {
    use vulkannn_rusted::cpu::ops::binary::sub::bf16;

    let mut group = c.benchmark_group("sub_bf16");
    for &n in SIZES {
        let a = make_bf16(n, 0x9E37_79B9);
        let b = make_bf16(n, 0x85EB_CA6B);
        let mut out = vec![half::bf16::ZERO; n];
        group.throughput(Throughput::Bytes((n * 2 * 3) as u64));

        // Tier II entry for bf16 (the generic kernel is private in its module).
        group.bench_with_input(BenchmarkId::new("dispatched", n), &n, |bencher, _| {
            bencher.iter(|| {
                bf16::sub_bf16(black_box(&a), black_box(&b), black_box(out.as_mut_slice()));
                black_box(out.len())
            })
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx") {
            group.bench_with_input(BenchmarkId::new("avx", n), &n, |bencher, _| {
                bencher.iter(|| {
                    unsafe {
                        bf16::sub_bf16_avx_serial(
                            black_box(&a),
                            black_box(&b),
                            black_box(out.as_mut_slice()),
                        )
                    };
                    black_box(out.len())
                })
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// dispatch / threshold overhead
// ---------------------------------------------------------------------------
/// Measures what the Tier II ladder itself costs, so later waves can tell a slow
/// kernel apart from a slow dispatcher.
fn bench_dispatch_overhead(c: &mut Criterion) {
    use vulkannn_rusted::cpu::dispatch;

    let mut group = c.benchmark_group("dispatch");
    group.bench_function("active_arch", |b| {
        b.iter(|| black_box(dispatch::active_arch()))
    });
    group.bench_function("detect_arch", |b| {
        b.iter(|| black_box(dispatch::detect_arch()))
    });
    group.finish();
}

criterion_group!(sub_f32, bench_sub_f32);
criterion_group!(add_f32, bench_add_f32);
criterion_group!(mul_f32, bench_mul_f32);
criterion_group!(div_f32, bench_div_f32);
criterion_group!(sum_f32, bench_sum_f32);
criterion_group!(atan2_f32, bench_atan2_f32);
criterion_group!(sub_bf16, bench_sub_bf16);
criterion_group!(dispatch_overhead, bench_dispatch_overhead);

criterion_main!(sub_f32, add_f32, mul_f32, div_f32, sum_f32, atan2_f32, sub_bf16, dispatch_overhead);

// sum (f64-accumulate reduction) — scalar naive f64 vs avx1 widen-accumulate.
fn bench_sum_f32(c: &mut Criterion) {
    use vulkannn_rusted::cpu::ops::reduction::sum::fp32;
    let mut group = c.benchmark_group("sum_f32");
    for &n in SIZES {
        let a = Aligned64::filled(n, 0x9E37_79B9);
        let a = a.as_slice();
        group.throughput(Throughput::Bytes((n * 4) as u64));
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| black_box(fp32::sum_f32_scalar::sum(black_box(a))))
        });
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx") {
            group.bench_with_input(BenchmarkId::new("avx1", n), &n, |b, _| {
                b.iter(|| black_box(unsafe { fp32::sum_f32_avx1::sum(black_box(a)) }))
            });
        }
    }
    group.finish();
}
