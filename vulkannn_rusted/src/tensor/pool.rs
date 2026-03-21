use std::cell::RefCell;

// Thread-local pool to avoid locking overhead on CPU-intensive critical paths.
thread_local! {
    pub static TENSOR_POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

pub struct TensorPool {
    // Buckets for different size classes.
    // 0: 4KB, 1: 64KB, 2: 1MB, 3: 4MB, 4: 16MB, 5: 64MB+
    buckets: Vec<Vec<Vec<u8>>>,
}

impl TensorPool {
    pub fn new() -> Self {
        let mut buckets = Vec::with_capacity(6);
        for _ in 0..6 {
            buckets.push(Vec::new());
        }
        Self { buckets }
    }

    fn get_bucket_idx(n_bytes: usize) -> Option<usize> {
        if n_bytes <= 4096 { Some(0) }
        else if n_bytes <= 65536 { Some(1) }
        else if n_bytes <= 1048576 { Some(2) }
        else if n_bytes <= 4194304 { Some(3) }
        else if n_bytes <= 16777216 { Some(4) }
        else if n_bytes <= 67108864 { Some(5) }
        else { None } // Too large for pooling
    }

    fn get_bucket_size(idx: usize) -> usize {
        match idx {
            0 => 4096,
            1 => 65536,
            2 => 1048576,
            3 => 4194304,
            4 => 16777216,
            5 => 67108864,
            _ => 0,
        }
    }

    pub fn alloc(&mut self, n_bytes: usize) -> Vec<u8> {
        if let Some(idx) = Self::get_bucket_idx(n_bytes) {
            if let Some(buf) = self.buckets[idx].pop() {
                return buf;
            }
            // Allocate new buffer of bucket size, ensuring 4-byte alignment
            // Allocate new buffer of bucket size, ensuring 8-byte alignment
            let bucket_size = Self::get_bucket_size(idx);
            let f64_count = (bucket_size + 7) / 8;
            let mut v = vec![0.0f64; f64_count];
            let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
            std::mem::forget(v);
            let v_u8 = unsafe { Vec::from_raw_parts(ptr as *mut u8, bucket_size, cap * 8) };
            return v_u8;
        }
        // Fallback to fresh allocation, ensuring 8-byte alignment and padding
        let f64_count = (n_bytes + 7) / 8;
        let padded_size = f64_count * 8;
        let mut v = vec![0.0f64; f64_count];
        let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
        std::mem::forget(v);
        let v_u8 = unsafe { Vec::from_raw_parts(ptr as *mut u8, padded_size, cap * 8) };
        return v_u8;
    }

    pub fn free(&mut self, buf: Vec<u8>) {
        let n_bytes = buf.len();
        if let Some(idx) = Self::get_bucket_idx(n_bytes) {
            // Only pool if it exactly matches a bucket size (to avoid fragmentation)
            if n_bytes == Self::get_bucket_size(idx) {
                // Keep pool size reasonable (e.g. max 16 buffers per bucket)
                if self.buckets[idx].len() < 16 {
                    self.buckets[idx].push(buf);
                    return;
                }
            }
        }
        // Implicitly drops buf if not returned to pool
    }
}
