use crate::tensor::{Tensor, DataType, Storage};

pub fn cat_f32(tensors: &[&Tensor], dim: usize) -> Vec<f32> {
    let mut out_shape = tensors[0].shape.clone();
    let total_dim: usize = tensors.iter().map(|t| t.shape[dim]).sum();
    out_shape[dim] = total_dim;
    
    let total_size: usize = out_shape.iter().product();
    let mut out_data = vec![0.0; total_size];
    
    let mut current_offset = 0;
    
    // Simplest case: dim 0 concatenation
    if dim == 0 {
        for t in tensors {
            if let Storage::F32(v) = &t.storage {
                out_data[current_offset..current_offset + v.len()].copy_from_slice(v);
                current_offset += v.len();
            }
        }
        return out_data;
    }
    
    // General case for inner dimensions
    // We iterate over the output shape and fill it.
    // Or more efficiently: copy chunks of contiguous memory.
    
    let outer_size: usize = out_shape[..dim].iter().product::<usize>();
    let inner_size: usize = out_shape[dim+1..].iter().product::<usize>();
    
    let out_stride_dim = total_dim * inner_size;
    
    for outer_idx in 0..outer_size {
        let mut dim_offset = 0;
        for t in tensors {
            let t_dim_size = t.shape[dim];
            let t_stride_dim = t_dim_size * inner_size;
            
            if let Storage::F32(v) = &t.storage {
                let src_start = outer_idx * t_stride_dim;
                let dst_start = outer_idx * out_stride_dim + dim_offset * inner_size;
                
                let len = t_stride_dim;
                out_data[dst_start..dst_start + len].copy_from_slice(&v[src_start..src_start + len]);
                dim_offset += t_dim_size;
            }
        }
    }
    
    out_data
}
