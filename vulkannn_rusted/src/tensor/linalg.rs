use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use super::{Tensor, DataType, Storage};

impl Tensor {
    pub fn execute_linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, activation: &str) -> PyResult<Tensor> {
        let m = input.shape[0];
        let k = input.shape[1];
        let n = weight.shape[0];
        let mut res = Tensor::new_zeros(vec![m, n], input.dtype, &input.device)?;

        if input.device == "cpu" {
            let (a, _) = input.get_slice_raw_f32();
            let (b, _) = weight.get_slice_raw_f32();
            let (c, _) = res.get_slice_raw_mut_f32();
            unsafe {
                matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), 1, k as isize, 0.0, c.as_mut_ptr(), n as isize, 1);
            }
            if let Some(b_t) = bias {
                let (bias_v, _) = b_t.get_slice_raw_f32();
                c.par_chunks_mut(n).for_each(|row| {
                    for j in 0..n { row[j] += bias_v[j]; }
                });
            }
            if activation == "relu" { crate::cpu::relu_f32_inplace(c); }
        } else {
             let (a_raw, _) = input.get_slice_raw_bytes();
             let (b_raw, _) = weight.get_slice_raw_bytes();
             let bias_raw = bias.map(|b| b.get_slice_raw_bytes().0).unwrap_or(&[]);
             let (out_raw, _) = res.get_slice_raw_mut_bytes();
             let act_type = match activation {
                 "relu" => 1,
                 "sigmoid" => 2,
                 _ => 0,
             };
             crate::backend::execute_linear_into(a_raw, b_raw, bias_raw, out_raw, m as u32, k as u32, n as u32, act_type, input.dtype);
        }
        Ok(res)
    }

    pub fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        Self::execute_linear(self, other, None, "none")
    }

    pub fn execute_transpose(&self) -> PyResult<Tensor> {
        if self.shape.len() != 2 { return Err(PyValueError::new_err("2D required for transpose")); }
        Ok(Tensor { 
            shape: vec![self.shape[1], self.shape[0]], 
            storage: self.storage.clone(), 
            dtype: self.dtype, 
            device: self.device.clone(), 
            name: format!("{}_T", self.name),
            is_transposed: !self.is_transposed,
            mmap_data: self.mmap_data.clone(),
        })
    }
}
