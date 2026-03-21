use crate::tensor::{Tensor, DataType, Storage};

pub fn cat_i8(tensors: &[&Tensor], dim: usize) -> Vec<i8> {
    debug_assert!(tensors[0].dtype == DataType::Int8 || tensors[0].dtype == DataType::Ternary);
    let mut out_shape = tensors[0].shape.clone();
    let total_dim: usize = tensors.iter().map(|t| t.shape[dim]).sum();
    out_shape[dim] = total_dim;
    let total_size: usize = out_shape.iter().product();
    let mut out_data = vec![0; total_size];
    
    let outer_size: usize = out_shape[..dim].iter().product::<usize>();
    let inner_size: usize = out_shape[dim+1..].iter().product::<usize>();
    let out_stride_dim = total_dim * inner_size;
    
    let mut dim_offsets = vec![0; tensors.len()];
    let mut current_offset = 0;
    for (i, t) in tensors.iter().enumerate() {
        dim_offsets[i] = current_offset;
        current_offset += t.shape[dim];
    }

    for outer_idx in 0..outer_size {
        for (i, t) in tensors.iter().enumerate() {
            let t_dim_size = t.shape[dim];
            let t_stride_dim = t_dim_size * inner_size;
            if let Storage::Int8(v) = &t.storage {
                let src_start = outer_idx * t_stride_dim;
                let dst_start = outer_idx * out_stride_dim + dim_offsets[i] * inner_size;
                out_data[dst_start..dst_start + t_stride_dim].copy_from_slice(&v[src_start..src_start + t_stride_dim]);
            } else if let Storage::Ternary(v) = &t.storage {
                let src_start = outer_idx * t_stride_dim;
                let dst_start = outer_idx * out_stride_dim + dim_offsets[i] * inner_size;
                out_data[dst_start..dst_start + t_stride_dim].copy_from_slice(&v[src_start..src_start + t_stride_dim]);
            }
        }
    }
    out_data
}
