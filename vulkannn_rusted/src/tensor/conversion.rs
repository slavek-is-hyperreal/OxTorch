use pyo3::prelude::*;
use rayon::prelude::*;
use super::{Tensor, DataType, Storage};

impl Tensor {
    pub fn execute_to_bitnet(&self, target_dtype: DataType) -> PyResult<Tensor> {
        if self.dtype != DataType::Int8 {
            return Err(pyo3::exceptions::PyValueError::new_err("Source must be Int8 for BitNet conversion"));
        }
        if target_dtype != DataType::BitNet2 && target_dtype != DataType::BitNet1_6 {
            return Err(pyo3::exceptions::PyValueError::new_err("Target must be BitNet2 or BitNet1_6"));
        }

        let (src_raw, _) = self.get_slice_raw_i8();
        let size = src_raw.len();
        
        let packed_data = if target_dtype == DataType::BitNet2 {
            // Row-interleaved 1x4 packing (LSB-first, matches safetensors format):
            // One byte at (row/4, col) stores weights for rows {row, row+1, row+2, row+3} at column col.
            // bits[1:0]=row0, bits[3:2]=row1, bits[5:4]=row2, bits[7:6]=row3
            let shape = &self.shape;
            let rows = shape[0];
            let cols = shape[1];
            assert!(rows % 4 == 0, "Wait: Row count must be multiple of 4 for interleaved packing");
            
            let n_bytes = (rows * cols) / 4;
            let mut packed = vec![0u8; n_bytes];
            
            // Parallelize over groups of 4 rows
            packed.par_chunks_mut(cols).enumerate().for_each(|(rg, row_group_slice)| {
                let r0 = rg * 4;
                let r1 = r0 + 1;
                let r2 = r0 + 2;
                let r3 = r0 + 3;
                
                for col in 0..cols {
                    let q0 = (src_raw[r0 * cols + col] + 1).clamp(0, 2) as u8;
                    let q1 = (src_raw[r1 * cols + col] + 1).clamp(0, 2) as u8;
                    let q2 = (src_raw[r2 * cols + col] + 1).clamp(0, 2) as u8;
                    let q3 = (src_raw[r3 * cols + col] + 1).clamp(0, 2) as u8;
                    
                    row_group_slice[col] = (q0 << 0) | (q1 << 2) | (q2 << 4) | (q3 << 6);
                }
            });
            packed
        } else {
            let n_bytes = (size + 4) / 5;
            let mut packed = vec![0u8; n_bytes];
            packed.par_chunks_mut(1).enumerate().for_each(|(i, byte_slice)| {
                let mut byte = 0u8;
                let mut power = 1u8;
                for j in 0..5 {
                    let idx = i * 5 + j;
                    if idx < size {
                        let val = (src_raw[idx] + 1).clamp(0, 2) as u8;
                        byte += val * power;
                        power *= 3;
                    }
                }
                byte_slice[0] = byte;
            });
            packed
        };

        let strides = Self::calculate_default_strides(self.shape.clone());
        Ok(Tensor {
            shape: self.shape.clone(),
            strides,
            offset: 0,
            device: self.device.clone(),
            dtype: target_dtype,
            storage: Storage::BitNet(packed_data),
            name: format!("{}_packed", self.name),
            is_transposed: false,
            mmap_data: None,
        })
    }
}
