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
use std::sync::{Mutex, OnceLock};
use half::{f16, bf16};

/// Maximum number of free buffers retained per size class.
const MAX_BUFS_PER_CLASS: usize = 4;
/// Number of power-of-2 size classes (2^0 through 2^30).
const NUM_CLASSES: usize = 31;

/// Internal pool storage.
struct Pool {
    buckets_f32: Vec<Mutex<Vec<Vec<f32>>>>,
    buckets_f16: Vec<Mutex<Vec<Vec<f16>>>>,
    buckets_bf16: Vec<Mutex<Vec<Vec<bf16>>>>,
    buckets_i8: Vec<Mutex<Vec<Vec<i8>>>>,
}

static INNER_POOL: OnceLock<Pool> = OnceLock::new();

fn inner() -> &'static Pool {
    INNER_POOL.get_or_init(|| {
        let mut buckets_f32 = Vec::with_capacity(NUM_CLASSES);
        let mut buckets_f16 = Vec::with_capacity(NUM_CLASSES);
        let mut buckets_bf16 = Vec::with_capacity(NUM_CLASSES);
        let mut buckets_i8 = Vec::with_capacity(NUM_CLASSES);
        for _ in 0..NUM_CLASSES {
            buckets_f32.push(Mutex::new(Vec::new()));
            buckets_f16.push(Mutex::new(Vec::new()));
            buckets_bf16.push(Mutex::new(Vec::new()));
            buckets_i8.push(Mutex::new(Vec::new()));
        }
        Pool { buckets_f32, buckets_f16, buckets_bf16, buckets_i8 }
    })
}

fn size_class(n: usize) -> usize {
    if n <= 1 { return 0; }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

pub struct BufPool;

impl BufPool {
    #[inline]
    pub fn get(n: usize) -> Vec<f32> {
        let cls = size_class(n).min(NUM_CLASSES - 1);
        if let Ok(mut bucket) = inner().buckets_f32[cls].try_lock() {
            if let Some(mut v) = bucket.pop() {
                v.resize(n, 0.0);
                return v;
            }
        }
        vec![0.0; n]
    }

    #[inline]
    pub fn put(mut v: Vec<f32>) {
        if v.capacity() == 0 { return; }
        let cls = size_class(v.capacity()).min(NUM_CLASSES - 1);
        v.clear();
        if let Ok(mut bucket) = inner().buckets_f32[cls].try_lock() {
            if bucket.len() < MAX_BUFS_PER_CLASS {
                bucket.push(v);
            }
        }
    }

    #[inline]
    pub fn get_f16(n: usize) -> Vec<f16> {
        let cls = size_class(n).min(NUM_CLASSES - 1);
        if let Ok(mut bucket) = inner().buckets_f16[cls].try_lock() {
            if let Some(mut v) = bucket.pop() {
                v.resize(n, f16::ZERO);
                return v;
            }
        }
        vec![f16::ZERO; n]
    }

    #[inline]
    pub fn put_f16(mut v: Vec<f16>) {
        if v.capacity() == 0 { return; }
        let cls = size_class(v.capacity()).min(NUM_CLASSES - 1);
        v.clear();
        if let Ok(mut bucket) = inner().buckets_f16[cls].try_lock() {
            if bucket.len() < MAX_BUFS_PER_CLASS {
                bucket.push(v);
            }
        }
    }

    #[inline]
    pub fn get_bf16(n: usize) -> Vec<bf16> {
        let cls = size_class(n).min(NUM_CLASSES - 1);
        if let Ok(mut bucket) = inner().buckets_bf16[cls].try_lock() {
            if let Some(mut v) = bucket.pop() {
                v.resize(n, bf16::ZERO);
                return v;
            }
        }
        vec![bf16::ZERO; n]
    }

    #[inline]
    pub fn put_bf16(mut v: Vec<bf16>) {
        if v.capacity() == 0 { return; }
        let cls = size_class(v.capacity()).min(NUM_CLASSES - 1);
        v.clear();
        if let Ok(mut bucket) = inner().buckets_bf16[cls].try_lock() {
            if bucket.len() < MAX_BUFS_PER_CLASS {
                bucket.push(v);
            }
        }
    }

    #[inline]
    pub fn get_i8(n: usize) -> Vec<i8> {
        let cls = size_class(n).min(NUM_CLASSES - 1);
        if let Ok(mut bucket) = inner().buckets_i8[cls].try_lock() {
            if let Some(mut v) = bucket.pop() {
                v.resize(n, 0);
                return v;
            }
        }
        vec![0; n]
    }

    #[inline]
    pub fn put_i8(mut v: Vec<i8>) {
        if v.capacity() == 0 { return; }
        let cls = size_class(v.capacity()).min(NUM_CLASSES - 1);
        v.clear();
        if let Ok(mut bucket) = inner().buckets_i8[cls].try_lock() {
            if bucket.len() < MAX_BUFS_PER_CLASS {
                bucket.push(v);
            }
        }
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
