use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use super::{Tensor, DataType};

impl Tensor {
    pub fn execute_linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, activation: &str) -> PyResult<Tensor> {
        let m = input.shape[0];
        let k = input.shape[1];
        let n = weight.shape[0];
        let mut res = Tensor::new_zeros(vec![m, n], input.dtype, &input.device)?;

        if input.device == "cpu" {
            match input.dtype {
                DataType::F32 => {
                    let (a, _) = input.get_slice_raw_f32();
                    let (b, _) = weight.get_slice_raw_f32();
                    let (c, _) = res.get_slice_raw_mut_f32();
                    crate::cpu::linear_f32(m as usize, k as usize, n as usize, a, b, c);
                },
                DataType::F16 => {
                    let (a, _) = input.get_slice_raw_f16();
                    let (b, _) = weight.get_slice_raw_f16();
                    let (c, _) = res.get_slice_raw_mut_f16();
                    crate::cpu::linear_f16(m, k, n, a, b, c);
                },
                DataType::BF16 => {
                    let (a, _) = input.get_slice_raw_bf16();
                    let (b, _) = weight.get_slice_raw_bf16();
                    let (c, _) = res.get_slice_raw_mut_bf16();
                    crate::cpu::linear_bf16(m, k, n, a, b, c);
                },
                _ => {
                    // Fallback for BF16/Int8
                    let a_f32 = input.to_numpy_f32_vec();
                    let b_f32 = weight.to_numpy_f32_vec();
                    let mut c_f32 = vec![0.0f32; (m * n) as usize];
                    crate::cpu::linear_f32(m as usize, k as usize, n as usize, &a_f32, &b_f32, &mut c_f32);
                    let res_dtype = res.dtype; let (res_bytes, _) = res.get_slice_raw_mut_bytes();
                    match res_dtype {
                        DataType::BF16 => crate::cpu::convert_f32_to_bf16(&c_f32, bytemuck::cast_slice_mut(res_bytes)),
                        DataType::Int8 => {
                             let dst = bytemuck::cast_slice_mut::<u8, i8>(res_bytes);
                             for i in 0..c_f32.len() { dst[i] = c_f32[i] as i8; }
                        }
                        _ => {}
                    }
                }
            }
            if let Some(b_t) = bias {
                let (bias_v, _) = b_t.get_slice_raw_f32();
                let (c, _) = res.get_slice_raw_mut_f32();
                c.par_chunks_mut(n as usize).for_each(|row| {
                    for j in 0..n as usize { row[j] += bias_v[j]; }
                });
            }
            if activation == "relu" { 
                let (c, _) = res.get_slice_raw_mut_f32();
                crate::cpu::relu_f32_inplace(c);
            }
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
             crate::backend::execute_linear_into(a_raw, b_raw, bias_raw, out_raw, m as u32, k as u32, n as u32, act_type, 1, input.dtype);
        }
        Ok(res)
    }

    pub fn bmm(&self, other: &Tensor) -> PyResult<Tensor> {
        if self.shape.len() != 3 || other.shape.len() != 3 {
             return Err(PyValueError::new_err("bmm requires 3D tensors"));
        }
        let b = self.shape[0];
        let m = self.shape[1];
        let k = self.shape[2];
        let n = other.shape[2];
        if other.shape[0] != b || other.shape[1] != k {
             return Err(PyValueError::new_err(format!("bmm shape mismatch: {:?} and {:?}", self.shape, other.shape)));
        }

        let mut res = Tensor::new_zeros(vec![b, m, n], self.dtype, &self.device)?;

        if self.device == "cpu" {
             // CPU Fallback: Loop over the batch dimension
             let a_f32 = self.to_numpy_f32_vec();
             let b_f32 = other.to_numpy_f32_vec();
             let mut c_f32 = vec![0.0f32; (b * m * n) as usize];
             
             let stride_a = (m * k) as usize;
             let stride_b = (k * n) as usize;
             let stride_c = (m * n) as usize;

             for i in 0..b as usize {
                 let a_slice = &a_f32[i * stride_a .. (i+1) * stride_a];
                 let b_slice = &b_f32[i * stride_b .. (i+1) * stride_b];
                 let c_slice = &mut c_f32[i * stride_c .. (i+1) * stride_c];
                 crate::cpu::matmul_f32(m as usize, k as usize, n as usize, a_slice, b_slice, c_slice);
             }

             // Write back the result bytes depending on output dtype
             let res_dtype = res.dtype; 
             let (res_bytes, _) = res.get_slice_raw_mut_bytes();
             match res_dtype {
                 DataType::F32 => bytemuck::cast_slice_mut::<u8, f32>(res_bytes).copy_from_slice(&c_f32),
                 DataType::F16 => crate::cpu::convert_f32_to_f16(&c_f32, bytemuck::cast_slice_mut(res_bytes)),
                 DataType::BF16 => crate::cpu::convert_f32_to_bf16(&c_f32, bytemuck::cast_slice_mut(res_bytes)),
                 DataType::Int8 | DataType::Ternary => {
                      let dst = bytemuck::cast_slice_mut::<u8, i8>(res_bytes);
                      for i in 0..c_f32.len() { dst[i] = c_f32[i] as i8; }
                 }
             }
        } else {
             // Vulkan single shader dispatch
             let (a_raw, _) = self.get_slice_raw_bytes();
             let (b_raw, _) = other.get_slice_raw_bytes();
             let (out_raw, _) = res.get_slice_raw_mut_bytes();
             crate::backend::execute_matmul_into(a_raw, b_raw, out_raw, b as u32, m as u32, k as u32, n as u32, self.dtype);


        }
        Ok(res)
    }

    pub fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(PyValueError::new_err("MatMul requires 2D tensors"));
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        if k != other.shape[0] {
            return Err(PyValueError::new_err(format!("MatMul shape mismatch: {:?} and {:?}", self.shape, other.shape)));
        }

        let mut res = Tensor::new_zeros(vec![m, n], self.dtype, &self.device)?;

        if self.device == "cpu" {
            match self.dtype {
                DataType::F32 => {
                    let (a, _) = self.get_slice_raw_f32();
                    let (b, _) = other.get_slice_raw_f32();
                    let (c, _) = res.get_slice_raw_mut_f32();
                    crate::cpu::matmul_f32(m as usize, k as usize, n as usize, a, b, c);
                },
                DataType::F16 => {
                    let (a, _) = self.get_slice_raw_f16();
                    let (b, _) = other.get_slice_raw_f16();
                    let (c, _) = res.get_slice_raw_mut_f16();
                    crate::cpu::matmul_f16(m as usize, k as usize, n as usize, a, b, c);
                },
                _ => {
                    let a_f32 = self.to_numpy_f32_vec();
                    let b_f32 = other.to_numpy_f32_vec();
                    let mut c_f32 = vec![0.0f32; (m * n) as usize];
                    crate::cpu::matmul_f32(m as usize, k as usize, n as usize, &a_f32, &b_f32, &mut c_f32);
                    let res_dtype = res.dtype; let (res_bytes, _) = res.get_slice_raw_mut_bytes();
                    match res_dtype {
                        DataType::BF16 => crate::cpu::convert_f32_to_bf16(&c_f32, bytemuck::cast_slice_mut(res_bytes)),
                        DataType::Int8 => {
                             let dst = bytemuck::cast_slice_mut::<u8, i8>(res_bytes);
                             for i in 0..c_f32.len() { dst[i] = c_f32[i] as i8; }
                        }
                        _ => {}
                    }
                }
            }
        } else {
             let (a_raw, _) = self.get_slice_raw_bytes();
             let (b_raw, _) = other.get_slice_raw_bytes();
             let (out_raw, _) = res.get_slice_raw_mut_bytes();
             crate::backend::execute_matmul_into(a_raw, b_raw, out_raw, 1, m as u32, k as u32, n as u32, self.dtype);
        }
        Ok(res)
    }

    pub fn layer_norm(&self, normalized_shape: Vec<usize>, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> PyResult<Tensor> {
        let mut d = 1;
        for dim in &normalized_shape { d *= dim; }
        let total = self.shape.iter().product::<usize>();
        if total % d != 0 {
            return Err(PyValueError::new_err("Invalid normalized_shape for layer_norm"));
        }
        let n = total / d;
        let mut res = Tensor::new_zeros(self.shape.clone(), self.dtype, &self.device)?;

        if self.device == "cpu" {
             match self.dtype {
                 DataType::F32 => {
                     let (x_raw, _) = self.get_slice_raw_f32();
                     let w_raw = if let Some(w) = weight { w.get_slice_raw_f32().0 } else { &[] };
                     let b_raw = if let Some(b) = bias   { b.get_slice_raw_f32().0 } else { &[] };
                     let (out_raw, _) = res.get_slice_raw_mut_f32();
                     crate::cpu::layer_norm_f32(x_raw, w_raw, b_raw, out_raw, n, d, eps);
                 },
                 DataType::F16 => {
                     let (x_raw, _) = self.get_slice_raw_f16();
                     let w_raw = if let Some(w) = weight { w.get_slice_raw_f16().0 } else { &[] };
                     let b_raw = if let Some(b) = bias   { b.get_slice_raw_f16().0 } else { &[] };
                     let (out_raw, _) = res.get_slice_raw_mut_f16();
                     crate::cpu::layer_norm_f16(x_raw, w_raw, b_raw, out_raw, n, d, eps);
                 },
                 DataType::BF16 => {
                     let (x_raw, _) = self.get_slice_raw_bf16();
                     let w_raw = if let Some(w) = weight { w.get_slice_raw_bf16().0 } else { &[] };
                     let b_raw = if let Some(b) = bias   { b.get_slice_raw_bf16().0 } else { &[] };
                     let (out_raw, _) = res.get_slice_raw_mut_bf16();
                     crate::cpu::layer_norm_bf16(x_raw, w_raw, b_raw, out_raw, n, d, eps);
                 },
                 _ => return Err(PyValueError::new_err("Unsupported dtype for layer_norm on CPU")),
             }
        } else {
             let (x_raw, _) = self.get_slice_raw_bytes();
             let w_raw = if let Some(w) = weight { w.get_slice_raw_bytes().0 } else { &[] };
             let b_raw = if let Some(b) = bias   { b.get_slice_raw_bytes().0 } else { &[] };
             let (out_raw, _) = res.get_slice_raw_mut_bytes();
             crate::backend::execute_layer_norm_into(x_raw, w_raw, b_raw, out_raw, n as u32, d as u32, eps, self.dtype);
        }
        Ok(res)
    }

    pub fn rms_norm(&self, normalized_shape: Vec<usize>, weight: Option<&Tensor>, eps: f32) -> PyResult<Tensor> {
        let mut d = 1;
        for dim in &normalized_shape { d *= dim; }
        let total = self.shape.iter().product::<usize>();
        if total % d != 0 {
            return Err(PyValueError::new_err("Invalid normalized_shape for rms_norm"));
        }
        let n = total / d;
        let mut res = Tensor::new_zeros(self.shape.clone(), self.dtype, &self.device)?;

        if self.device == "cpu" {
             match self.dtype {
                 DataType::F32 => {
                     let (x_raw, _) = self.get_slice_raw_f32();
                     let w_raw = if let Some(w) = weight { w.get_slice_raw_f32().0 } else { &[] };
                     let (out_raw, _) = res.get_slice_raw_mut_f32();
                     crate::cpu::rms_norm_f32(x_raw, w_raw, out_raw, n, d, eps);
                 },
                 DataType::F16 => {
                     let (x_raw, _) = self.get_slice_raw_f16();
                     let w_raw = if let Some(w) = weight { w.get_slice_raw_f16().0 } else { &[] };
                     let (out_raw, _) = res.get_slice_raw_mut_f16();
                     crate::cpu::rms_norm_f16(x_raw, w_raw, out_raw, n, d, eps);
                 },
                 DataType::BF16 => {
                     let (x_raw, _) = self.get_slice_raw_bf16();
                     let w_raw = if let Some(w) = weight { w.get_slice_raw_bf16().0 } else { &[] };
                     let (out_raw, _) = res.get_slice_raw_mut_bf16();
                     crate::cpu::rms_norm_bf16(x_raw, w_raw, out_raw, n, d, eps);
                 },
                 _ => return Err(PyValueError::new_err("Unsupported dtype for rms_norm on CPU")),
             }
        } else {
             let (x_raw, _) = self.get_slice_raw_bytes();
             let w_raw = if let Some(w) = weight { w.get_slice_raw_bytes().0 } else { &[] };
             let (out_raw, _) = res.get_slice_raw_mut_bytes();
             crate::backend::execute_rms_norm_into(x_raw, w_raw, out_raw, n as u32, d as u32, eps, self.dtype);
        }
        Ok(res)
    }

    pub fn execute_bit_linear(input: &Tensor, weight: &Tensor, scale: &Tensor, bias: Option<&Tensor>) -> PyResult<Tensor> {
        let m = input.shape[0];
        let k = input.shape[1];
        let n = weight.shape[0];
        
        if input.dtype != DataType::Int8 { return Err(PyValueError::new_err("BitLinear input must be Int8")); }
        if weight.dtype != DataType::Ternary { return Err(PyValueError::new_err("BitLinear weights must be Ternary")); }
        if weight.shape[1] != k { return Err(PyValueError::new_err("BitLinear shape mismatch (K dimension)")); }
        
        // Result is dequantized to F32 (or could be F16/Int8 target)
        let mut res = Tensor::new_zeros(vec![m, n], DataType::F32, &input.device)?;
        
        if input.device == "cpu" {
            let (a, _) = input.get_slice_raw_i8();
            let (b, _) = weight.get_slice_raw_ternary();
            let (s, _) = scale.get_slice_raw_f32();
            let (c, _) = res.get_slice_raw_mut_f32();
            
            crate::cpu::bit_linear_f32(m, k, n, a, b, s, c);
            
            if let Some(b_t) = bias {
                let (bias_v, _) = b_t.get_slice_raw_f32();
                c.par_chunks_mut(n).for_each(|row| {
                    for j in 0..n { row[j] += bias_v[j]; }
                });
            }
        } else {
            let (a_raw, _) = input.get_slice_raw_bytes();
            let (b_raw, _) = weight.get_slice_raw_bytes();
            let (s_raw, _) = scale.get_slice_raw_bytes();
            let bias_raw = bias.map(|b| b.get_slice_raw_bytes().0).unwrap_or(&[]);
            let (out_raw, _) = res.get_slice_raw_mut_bytes();
            
            crate::backend::execute_bit_linear_into(a_raw, b_raw, s_raw, bias_raw, out_raw, m as u32, k as u32, n as u32);
        }
        
        Ok(res)
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
