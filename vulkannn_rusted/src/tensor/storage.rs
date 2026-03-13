use crate::buf_pool::BufPool;

/// Backbone data storage for the Tensor engine.
/// Encapsulates the multi-precision vectors (F32, F16, BF16) or SSD-mapped handles.
#[derive(Clone)]
pub enum Storage {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    BF16(Vec<half::bf16>),
    Int8(Vec<i8>),
    None,
}

impl Drop for Storage {
    fn drop(&mut self) {
        match self {
            Storage::F32(v) => BufPool::put(std::mem::take(v)),
            Storage::F16(v) => BufPool::put_f16(std::mem::take(v)),
            Storage::BF16(v) => BufPool::put_bf16(std::mem::take(v)),
            Storage::Int8(v) => BufPool::put_i8(std::mem::take(v)),
            _ => {}
        }
    }
}
