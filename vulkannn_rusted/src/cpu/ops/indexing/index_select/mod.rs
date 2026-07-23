//! INDEX_SELECT — gather rows of `weight` (a [num_rows, feature_len] table) at
//! `indices` into `out`. This is the embedding lookup. Indices are i32 (the
//! tensor layer converts torch's i64 -> i32 at the boundary).
//!
//! Pure memory gather (per-row memcpy), bit-exact — no numeric error, so the
//! parity criterion is bitwise equality (a non-bit difference would be a bug,
//! not a tolerance issue). Implemented with `copy_nonoverlapping` per row, which
//! the compiler lowers to an optimal (SIMD/`rep movsb`) memcpy — equivalent to
//! and typically at least as fast as legacy's hand-rolled AVX copy loop, without
//! the per-tier duplication (§8: memory-bound, don't over-invest).
//!
//! OUT-OF-RANGE: legacy did an UNCHECKED `ptr.add(idx*feature_len)` (UB on an
//! out-of-bounds index). We do NOT transcribe that UB — every index is validated
//! up front; an out-of-range (or negative) index panics, which pyo3 surfaces as a
//! Python exception, matching torch's error behaviour. Duplicate indices are
//! valid (common in embeddings) and just gather the same row twice.
//!
//! embedding(): index_select IS the embedding gather; there is no separate
//! `embedding` op in legacy. A Python-facing `embedding` (adding the embedding_dim
//! axis) would be a thin wrapper over this; `padding_idx` is NOT supported here
//! (documented, not silently dropped) — a follow-up if needed.

#[inline]
fn validate(indices: &[i32], weight_len: usize, feature_len: usize) {
    let num_rows = if feature_len == 0 { 0 } else { weight_len / feature_len };
    for &idx in indices {
        assert!(
            idx >= 0 && (idx as usize) < num_rows,
            "index_select: index {idx} out of range for {num_rows} rows"
        );
    }
}

macro_rules! index_select {
    ($fn:ident, $ty:ty) => {
        pub fn $fn(weight: &[$ty], indices: &[i32], out: &mut [$ty], feature_len: usize) {
            validate(indices, weight.len(), feature_len);
            for (i, &idx) in indices.iter().enumerate() {
                let src = (idx as usize) * feature_len;
                let dst = i * feature_len;
                // Safe: validate() guarantees src+feature_len <= weight.len(),
                // and out is sized num_indices*feature_len by the caller.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        weight.as_ptr().add(src),
                        out.as_mut_ptr().add(dst),
                        feature_len,
                    );
                }
            }
        }
    };
}

index_select!(index_select_f32, f32);
index_select!(index_select_f16, half::f16);
index_select!(index_select_bf16, half::bf16);
index_select!(index_select_i8, i8);

/// Thin embedding wrapper = index_select over an [vocab, dim] table.
/// padding_idx is NOT supported (see module note). Kept public so a Python-facing
/// embedding can call it without duplicating the gather.
pub fn embedding_f32(weight: &[f32], ids: &[i32], out: &mut [f32], embedding_dim: usize) {
    index_select_f32(weight, ids, out, embedding_dim);
}

#[cfg(test)]
mod t {
    use super::*;
    #[test]
    fn gather_and_duplicates() {
        // 4 rows x 2 features
        let w = [10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0];
        let idx = [2i32, 0, 2, 3]; // duplicate row 2
        let mut out = [0f32; 8];
        index_select_f32(&w, &idx, &mut out, 2);
        assert_eq!(out, [30.0, 31.0, 10.0, 11.0, 30.0, 31.0, 40.0, 41.0]);
    }
    #[test]
    #[should_panic(expected = "out of range")]
    fn out_of_range_panics() {
        let w = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0f32; 2];
        index_select_f32(&w, &[5i32], &mut out, 2); // row 5 in a 2-row table
    }
    #[test]
    #[should_panic(expected = "out of range")]
    fn negative_index_panics() {
        let w = [1.0f32, 2.0];
        let mut out = [0f32; 2];
        index_select_f32(&w, &[-1i32], &mut out, 2);
    }
}
