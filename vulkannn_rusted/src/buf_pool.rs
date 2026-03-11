//! buf_pool.rs — Free-list buffer pool for f32 Vec reuse.
//!
//! Architecture (per deep_research_on_optimization.md §6.2 "Zero-Copy Mappings"):
//! Instead of allocating a fresh Vec<f32> on every relu()/sigmoid()/etc call and
//! immediately freeing it, we park the backing memory in a size-class bucket and
//! hand it back on the next call of the same size class.
//!
//! - Size classes: bucketed by nearest power-of-2 ceiling of element count.
//!   Bucket index = ceil_log2(n).  We cover 2^0 = 1 .. 2^30 = ~1G elements.
//! - Per-bucket free-list capped at MAX_BUFS to bound memory usage.
//! - Thread safety: try_lock per bucket (non-blocking — graceful degradation).
//!
//! Usage in tensor.rs:
//!   let mut v = BufPool::get(n);   // pop from pool or alloc fresh
//!   v.resize(n, 0.0f32);          // ensure correct length (pool buf may be larger)
//!   // ... fill v ...
//!   // Either return result (let pool pick up on next get), OR:
//!   BufPool::put(v);               // explicitly return to pool

use std::sync::{Mutex, OnceLock};

/// Maximum number of free buffers retained per size class.
const MAX_BUFS_PER_CLASS: usize = 4;
/// Number of power-of-2 size classes (2^0 through 2^30).
const NUM_CLASSES: usize = 31;

/// Internal pool storage.
struct Pool {
    /// buckets[i] holds Vec<f32> buffers whose capacity rounds up to 2^i elements.
    buckets: Vec<Mutex<Vec<Vec<f32>>>>,
}

static INNER_POOL: OnceLock<Pool> = OnceLock::new();

fn inner() -> &'static Pool {
    INNER_POOL.get_or_init(|| {
        let mut buckets = Vec::with_capacity(NUM_CLASSES);
        for _ in 0..NUM_CLASSES {
            buckets.push(Mutex::new(Vec::new()));
        }
        Pool { buckets }
    })
}

/// Returns the power-of-2 ceiling log2 of n (size class index).
#[inline]
fn size_class(n: usize) -> usize {
    if n <= 1 { return 0; }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// Public API — zero-sized struct (all state is in the global `INNER_POOL`).
pub struct BufPool;

impl BufPool {
    /// Obtain a `Vec<f32>` with at least `n` elements (may be longer).
    /// Call `.resize(n, 0.0f32)` after to ensure exact length.
    /// Returns a pooled buffer if available, otherwise allocates fresh.
    #[inline]
    pub fn get(n: usize) -> Vec<f32> {
        let cls = size_class(n).min(NUM_CLASSES - 1);
        if let Ok(mut bucket) = inner().buckets[cls].try_lock() {
            if let Some(mut v) = bucket.pop() {
                // Pooled buffer reuse: warm memory, no OS page fault
                v.resize(n, 0.0f32);
                return v;
            }
        }
        // Fresh allocation (first call for this size class)
        vec![0.0f32; n]
    }

    /// Return a `Vec<f32>` to the pool for future reuse.
    /// Silently drops if the bucket is full or lock is contended.
    #[inline]
    pub fn put(mut v: Vec<f32>) {
        if v.capacity() == 0 { return; }
        let cls = size_class(v.capacity()).min(NUM_CLASSES - 1);
        v.clear(); // length → 0, capacity stays
        if let Ok(mut bucket) = inner().buckets[cls].try_lock() {
            if bucket.len() < MAX_BUFS_PER_CLASS {
                bucket.push(v);
            }
            // else: bucket full, Vec drops and OS reclaims
        }
        // lock contended: Vec drops, that's fine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_pool_round_trip() {
        let v = BufPool::get(1000);
        assert_eq!(v.len(), 1000);
        BufPool::put(v);
        let v2 = BufPool::get(1000);
        assert_eq!(v2.len(), 1000);
    }

    #[test]
    fn test_size_class() {
        assert_eq!(size_class(1), 0);
        assert_eq!(size_class(2), 1);
        assert_eq!(size_class(4), 2);
        assert_eq!(size_class(1_000_000), 20); // ceil_log2(1M) = 20
    }
}
