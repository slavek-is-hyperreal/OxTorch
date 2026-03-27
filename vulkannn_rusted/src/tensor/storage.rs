

/// Backbone data storage for the Tensor engine.
/// Encapsulates the multi-precision vectors (F32, F16, BF16) or SSD-mapped handles.
#[derive(Clone)]
pub enum Storage {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    BF16(Vec<half::bf16>),
    Int8(Vec<i8>),
    BitNet(Vec<u8>), // Packed BitNet weights (BitNet2 or BitNet1_6)
    #[allow(non_camel_case_types)]
    I2_S(Vec<u8>),   // Packed GGML I2_S chunks
    None,
}

impl Storage {
    pub fn as_bf16(&self) -> Option<&[half::bf16]> {
        if let Storage::BF16(v) = self { Some(v) } else { None }
    }
    pub fn as_bf16_mut(&mut self) -> Option<&mut [half::bf16]> {
        if let Storage::BF16(v) = self { Some(v) } else { None }
    }
    pub fn as_f32(&self) -> Option<&[f32]> {
        if let Storage::F32(v) = self { Some(v) } else { None }
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if let Storage::None = self { return; }
        
        let old = std::mem::replace(self, Storage::None);
        let mut old = std::mem::ManuallyDrop::new(old);
        
        // Safety: We are essentially doing what bytemuck::cast_vec does but manually.
        // All types (f32, f16, bf16, i8) are PoD and their alignments are 
        // satisfied by the pooled Vec<u8> (which we ensure is at least 8-byte aligned).
        let (buf, ptr_val) = match &mut *old {
            Storage::F32(v) => unsafe {
                let mut v = std::mem::take(v);
                let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                std::mem::forget(v);
                (Vec::from_raw_parts(ptr as *mut u8, cap * 4, cap * 4), ptr as usize)
            },
            Storage::F16(v) => unsafe {
                let mut v = std::mem::take(v);
                let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                std::mem::forget(v);
                (Vec::from_raw_parts(ptr as *mut u8, cap * 2, cap * 2), ptr as usize)
            },
            Storage::BF16(v) => unsafe {
                let mut v = std::mem::take(v);
                let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                std::mem::forget(v);
                (Vec::from_raw_parts(ptr as *mut u8, cap * 2, cap * 2), ptr as usize)
            },
            Storage::Int8(v) => unsafe {
                let mut v = std::mem::take(v);
                let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                std::mem::forget(v);
                (Vec::from_raw_parts(ptr as *mut u8, cap, cap), ptr as usize)
            },
            Storage::BitNet(v) => unsafe {
                let mut v = std::mem::take(v);
                let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                std::mem::forget(v);
                (Vec::from_raw_parts(ptr as *mut u8, cap, cap), ptr as usize)
            },
            Storage::I2_S(v) => unsafe {
                let mut v = std::mem::take(v);
                let (ptr, _len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                std::mem::forget(v);
                (Vec::from_raw_parts(ptr as *mut u8, cap, cap), ptr as usize)
            },
            Storage::None => return,
        };
        
        // Critical: Only return to pool if the original pointer is 8-byte aligned.
        // This ensures that when the buffer is pulled for any typed tensor later,
        // it satisfies the alignment requirement of the destination type (up to f64).
        if ptr_val % 8 == 0 {
            super::pool::TENSOR_POOL.with(|pool| {
                pool.borrow_mut().free(buf);
            });
        }
    }
}
