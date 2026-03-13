use pyo3::prelude::*;
use super::Tensor;

impl Tensor {
    pub fn execute_reshape(&self, new_shape: Vec<usize>) -> PyResult<Tensor> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        if old_size != new_size { return Err(pyo3::exceptions::PyValueError::new_err(format!("Reshape size mismatch: {} vs {}", old_size, new_size))); }
        Ok(Tensor { 
            shape: new_shape, 
            storage: self.storage.clone(), 
            dtype: self.dtype, 
            device: self.device.clone(), 
            name: format!("{}_reshaped", self.name),
            is_transposed: self.is_transposed,
            mmap_data: self.mmap_data.clone(),
        })
    }

    pub fn elementwise_op(&self, other: &Tensor, op: &str) -> PyResult<Tensor> {
        if self.shape != other.shape { return Err(pyo3::exceptions::PyValueError::new_err("Shapes must match")); }
        let mut out = Tensor::new_zeros(self.shape.clone(), self.dtype, &self.device)?;
        if self.device == "cpu" {
             let (a, _) = self.get_slice_raw_f32();
             let (b, _) = other.get_slice_raw_f32();
             let (res, _) = out.get_slice_raw_mut_f32();
             match op {
                 "add" => { for i in 0..a.len() { res[i] = a[i] + b[i]; } },
                 "sub" => { for i in 0..a.len() { res[i] = a[i] - b[i]; } },
                 "mul" => { for i in 0..a.len() { res[i] = a[i] * b[i]; } },
                 "div" => { for i in 0..a.len() { res[i] = a[i] / b[i]; } },
                 _ => {},
             }
        } else {
            let (a_raw, _) = self.get_slice_raw_bytes();
            let (b_raw, _) = other.get_slice_raw_bytes();
            let (out_raw, _) = out.get_slice_raw_mut_bytes();
            let op_id = match op {
                "mul" => 0,
                "sub" => 1,
                "div" => 2,
                "add" => 3,
                _ => 3,
            };
            crate::backend::execute_elementwise_into(a_raw, b_raw, out_raw, op_id, self.dtype);
        }
        Ok(out)
    }

    pub fn unary_op(&self, op: &str, param1: f32, param2: f32) -> PyResult<Tensor> {
        if self.is_ssd() { return self.unary_op_ssd(op, param1, param2); }
        let mut out = Tensor::new_zeros(self.shape.clone(), self.dtype, &self.device)?;
        if self.device == "cpu" {
            let (v_in, _) = self.get_slice_raw_f32();
            let (v_out, _) = out.get_slice_raw_mut_f32();
            match op {
                "relu" => crate::cpu::relu_f32(v_in, v_out),
                "gelu" => { v_out.copy_from_slice(v_in); crate::cpu::gelu_f32_inplace(v_out); },
                _ => { v_out.copy_from_slice(v_in); },
            }
        } else {
            let (input_raw, _) = self.get_slice_raw_bytes();
            let (out_raw, _) = out.get_slice_raw_mut_bytes();
            crate::backend::execute_activation_into(input_raw, op, param1, param2, out_raw, self.dtype, self.device == "hybrid", false);
        }
        Ok(out)
    }

    pub fn act_into_raw_parallel_f32(slice: &mut [f32], op: &str, _param1: f32, _param2: f32) {
        match op {
            "relu" => crate::cpu::relu_f32_inplace(slice),
            "gelu" => crate::cpu::gelu_f32_inplace(slice),
            _ => {},
        }
    }
}
