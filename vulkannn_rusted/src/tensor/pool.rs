use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::alloc::{alloc, dealloc, Layout};
use std::mem::{size_of, forget};

// Thread-local pool to avoid locking overhead on CPU-intensive critical paths.
thread_local! {
    pub static TENSOR_POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

/// Generic Buffer Guard for any POD (Plain Old Data) type.
/// Part of the OxTorch Scientific-Grade Specialization Matrix.
pub struct BufferGuard<T> {
    vec: Option<Vec<T>>,
}

impl<T> Deref for BufferGuard<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target { self.vec.as_ref().unwrap() }
}

impl<T> DerefMut for BufferGuard<T> {
    fn deref_mut(&mut self) -> &mut [T] { self.vec.as_mut().unwrap() }
}

impl<T> Drop for BufferGuard<T> {
    fn drop(&mut self) {
        if let Some(v) = self.vec.take() {
            TENSOR_POOL.with(|pool| pool.borrow_mut().free(v));
        }
    }
}

pub struct TensorPool {
    /// Logarithmic Buckets: buckets[i] contains buffers of capacity 2^i bytes.
    /// We support 2^10 (1KB) to 2^31 (2GB).
    buckets: Vec<Vec<Vec<u8>>>,
}

impl TensorPool {
    pub fn new() -> Self {
        let mut buckets = Vec::with_capacity(32);
        for _ in 0..32 {
            buckets.push(Vec::new());
        }
        Self { buckets }
    }

    /// Allocates any POD buffer from the pool with 64-byte alignment.
    pub fn get_buffer<T>(n_elems: usize) -> BufferGuard<T> {
        let n_bytes = n_elems * size_of::<T>();
        
        TENSOR_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            
            // Find smallest power-of-2 bucket that fits
            let target_size = n_bytes.next_power_of_two().max(1024);
            let idx = target_size.trailing_zeros() as usize;

            if idx < p.buckets.len() {
                if let Some(buf_u8) = p.buckets[idx].pop() {
                    let ptr = buf_u8.as_ptr();
                    let cap = buf_u8.capacity();
                    forget(buf_u8);
                    
                    let res_vec = unsafe { Vec::from_raw_parts(ptr as *mut T, n_elems, cap / size_of::<T>()) };
                    return BufferGuard { vec: Some(res_vec) };
                }
            }

            // Fallback: Allocate fresh aligned memory
            let buf_u8 = Self::alloc_aligned_raw(target_size);
            let ptr = buf_u8.as_ptr();
            let cap = buf_u8.capacity();
            forget(buf_u8);
            
            let res_vec = unsafe { Vec::from_raw_parts(ptr as *mut T, n_elems, cap / size_of::<T>()) };
            BufferGuard { vec: Some(res_vec) }
        })
    }

    /// Internal: Allocates a 64-byte aligned block of memory.
    fn alloc_aligned_raw(size: usize) -> Vec<u8> {
        let layout = Layout::from_size_align(size, 64).expect("Invalid layout");
        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Vec::from_raw_parts(ptr, size, size)
        }
    }

    /// Internal: Deallocates a 64-byte aligned block of memory.
    fn dealloc_aligned_raw(buf: Vec<u8>) {
        let size = buf.capacity();
        let ptr = buf.as_ptr();
        let layout = Layout::from_size_align(size, 64).expect("Invalid layout");
        forget(buf);
        unsafe { dealloc(ptr as *mut u8, layout); }
    }

    /// Returns a buffer to the pool.
    pub fn free<T>(&mut self, mut vec: Vec<T>) {
        let n_bytes = vec.capacity() * size_of::<T>();
        let ptr = vec.as_mut_ptr();
        forget(vec);

        let buf_u8 = unsafe { Vec::from_raw_parts(ptr as *mut u8, n_bytes, n_bytes) };

        // Only pool if it's a perfect power-of-2 and fits in our buckets.
        if n_bytes >= 1024 && n_bytes.is_power_of_two() {
            let idx = n_bytes.trailing_zeros() as usize;
            if idx < self.buckets.len() {
                if self.buckets[idx].len() < 64 { // Deep pool for heavy MSTS loads
                    self.buckets[idx].push(buf_u8);
                    return;
                }
            }
        }
        
        Self::dealloc_aligned_raw(buf_u8);
    }
    
    /// Legacy/Byte-level raw access
    pub fn alloc_raw(&mut self, n_bytes: usize) -> Vec<u8> {
        let target_size = n_bytes.next_power_of_two().max(1024);
        let idx = target_size.trailing_zeros() as usize;
        
        if idx < self.buckets.len() {
            if let Some(buf) = self.buckets[idx].pop() {
                return buf;
            }
        }
        Self::alloc_aligned_raw(target_size)
    }

    pub fn free_raw(&mut self, buf: Vec<u8>) {
        let n_bytes = buf.capacity();
        if n_bytes >= 1024 && n_bytes.is_power_of_two() {
            let idx = n_bytes.trailing_zeros() as usize;
            if idx < self.buckets.len() {
                if self.buckets[idx].len() < 64 {
                    self.buckets[idx].push(buf);
                    return;
                }
            }
        }
        Self::dealloc_aligned_raw(buf);
    }
}

impl Drop for TensorPool {
    fn drop(&mut self) {
        for bucket in &mut self.buckets {
            while let Some(buf) = bucket.pop() {
                Self::dealloc_aligned_raw(buf);
            }
        }
    }
}
