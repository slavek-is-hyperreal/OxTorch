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
            let n_bytes = (size + 3) / 4;
            let mut packed = vec![0u8; n_bytes];
            packed.par_chunks_mut(1).enumerate().for_each(|(i, byte_slice)| {
                let mut byte = 0u8;
                for j in 0..4 {
                    let idx = i * 4 + j;
                    if idx < size {
                        let val = (src_raw[idx] + 1).clamp(0, 2) as u8;
                        byte |= val << (j * 2);
                    }
                }
                byte_slice[0] = byte;
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
