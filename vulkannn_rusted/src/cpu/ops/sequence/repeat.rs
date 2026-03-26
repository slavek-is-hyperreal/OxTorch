use crate::tensor::DataType;

/// Repeat interleave for tensors.
/// Repeats each element of the tensor `repeats` times along `dim`.
pub fn repeat_interleave_cpu(
    in_buf: &[u8],
    out_buf: &mut [u8],
    shape: &[usize],
    repeats: usize,
    dim: usize,
    dtype: DataType,
) {
    let elem_size = dtype.size();
    let mut stride = 1;
    for i in dim..shape.len() { stride *= shape[i]; }
    let outer = shape[..dim].iter().product::<usize>();
    let dim_size = shape[dim];
    let inner = stride / dim_size;
    
    let chunk_size = inner * elem_size;
    
    for i in 0..outer {
        for j in 0..dim_size {
            let src_idx = (i * dim_size + j) * chunk_size;
            let src_chunk = &in_buf[src_idx .. src_idx + chunk_size];
            
            for r in 0..repeats {
                let dst_idx = (i * dim_size * repeats + j * repeats + r) * chunk_size;
                out_buf[dst_idx .. dst_idx + chunk_size].copy_from_slice(src_chunk);
            }
        }
    }
}
