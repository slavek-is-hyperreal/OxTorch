use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

// Thread-local pool to avoid locking overhead on CPU-intensive critical paths.
thread_local! {
    pub static TENSOR_POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

pub struct BufferGuardF32 {
    vec: Option<Vec<f32>>,
}

impl Deref for BufferGuardF32 {
    type Target = [f32];
    fn deref(&self) -> &Self::Target { self.vec.as_ref().unwrap() }
}

impl DerefMut for BufferGuardF32 {
    fn deref_mut(&mut self) -> &mut Self::Target { self.vec.as_mut().unwrap() }
}

impl Drop for BufferGuardF32 {
    fn drop(&mut self) {
        if let Some(v) = self.vec.take() {
            TENSOR_POOL.with(|pool| pool.borrow_mut().free_f32(v));
        }
    }
}

pub struct TensorPool {
    // Buckets for different size classes.
    // 0: 4KB, 1: 64KB, 2: 1MB, 3: 4MB, 4: 16MB, 5: 64MB+
    u8_buckets: Vec<Vec<Vec<u8>>>,
    f32_buckets: Vec<Vec<Vec<f32>>>,
}

impl TensorPool {
    pub fn new() -> Self {
        let mut u8_buckets = Vec::with_capacity(6);
        let mut f32_buckets = Vec::with_capacity(6);
        for _ in 0..6 {
            u8_buckets.push(Vec::new());
            f32_buckets.push(Vec::new());
        }
        Self { u8_buckets, f32_buckets }
    }

    fn get_bucket_idx(n_bytes: usize) -> Option<usize> {
        if n_bytes <= 4096 { Some(0) }
        else if n_bytes <= 65536 { Some(1) }
        else if n_bytes <= 1048576 { Some(2) }
        else if n_bytes <= 4194304 { Some(3) }
        else if n_bytes <= 16777216 { Some(4) }
        else if n_bytes <= 67108864 { Some(5) }
        else { None }
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

    pub fn alloc_f32(n_elems: usize) -> Vec<f32> {
        TENSOR_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            let n_bytes = n_elems * 4;
            if let Some(idx) = Self::get_bucket_idx(n_bytes) {
                if let Some(mut buf) = p.f32_buckets[idx].pop() {
                    unsafe { buf.set_len(n_elems); }
                    return buf;
                }
                let bucket_size = Self::get_bucket_size(idx);
                let f64_count = (bucket_size + 7) / 8;
                let mut v = vec![0.0f64; f64_count];
                let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                std::mem::forget(v);
                let v_f32 = unsafe { Vec::from_raw_parts(ptr as *mut f32, n_elems, cap * 2) };
                return v_f32;
            }
            vec![0.0f32; n_elems]
        })
    }

    pub fn get_f32_buffer(n_elems: usize) -> BufferGuardF32 {
        BufferGuardF32 { vec: Some(Self::alloc_f32(n_elems)) }
    }

    pub fn free_f32(&mut self, mut buf: Vec<f32>) {
        let n_bytes = buf.capacity() * 4;
        if let Some(idx) = Self::get_bucket_idx(n_bytes) {
            if n_bytes == Self::get_bucket_size(idx) {
                if self.f32_buckets[idx].len() < 16 {
                    unsafe { buf.set_len(0); }
                    self.f32_buckets[idx].push(buf);
                    return;
                }
            }
        }
    }

    pub fn alloc(&mut self, n_bytes: usize) -> Vec<u8> {
        if let Some(idx) = Self::get_bucket_idx(n_bytes) {
            if let Some(buf) = self.u8_buckets[idx].pop() {
                return buf;
            }
            let bucket_size = Self::get_bucket_size(idx);
            let f64_count = (bucket_size + 7) / 8;
            let mut v = vec![0.0f64; f64_count];
            let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
            std::mem::forget(v);
            let v_u8 = unsafe { Vec::from_raw_parts(ptr as *mut u8, bucket_size, cap * 8) };
            return v_u8;
        }
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
            if n_bytes == Self::get_bucket_size(idx) {
                if self.u8_buckets[idx].len() < 16 {
                    self.u8_buckets[idx].push(buf);
                    return;
                }
            }
        }
    }
}
