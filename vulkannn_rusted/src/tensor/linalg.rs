use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::tensor::{Tensor, DataType, Storage};
use rayon::prelude::*;

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
             
             let stride_a = (m * k) as usize;
             let stride_b = (k * n) as usize;
             let stride_c = (m * n) as usize;

             let res_dtype = res.dtype;
             let (res_bytes, _) = res.get_slice_raw_mut_bytes();

             if res_dtype == DataType::F32 {
                 let c_f32: &mut [f32] = bytemuck::cast_slice_mut(res_bytes);
                 for i in 0..b as usize {
                     let a_slice = &a_f32[i * stride_a .. (i+1) * stride_a];
                     let b_slice = &b_f32[i * stride_b .. (i+1) * stride_b];
                     let c_slice = &mut c_f32[i * stride_c .. (i+1) * stride_c];
                     crate::cpu::matmul_f32(m as usize, k as usize, n as usize, a_slice, b_slice, c_slice);
                 }
             } else {
                 let total_elems = (b * m * n) as usize;
                 let mut c_f32_vec = super::pool::TENSOR_POOL.with(|pool| {
                     let mut p = pool.borrow_mut();
                     let mut buf = p.alloc(total_elems * 4);
                     let (ptr, _len, cap) = (buf.as_mut_ptr(), buf.len(), buf.capacity());
                     std::mem::forget(buf);
                     unsafe { Vec::from_raw_parts(ptr as *mut f32, total_elems, cap / 4) }
                 });

                 for i in 0..b as usize {
                     let a_slice = &a_f32[i * stride_a .. (i+1) * stride_a];
                     let b_slice = &b_f32[i * stride_b .. (i+1) * stride_b];
                     let c_slice = &mut c_f32_vec[i * stride_c .. (i+1) * stride_c];
                     crate::cpu::matmul_f32(m as usize, k as usize, n as usize, a_slice, b_slice, c_slice);
                 }

                 match res_dtype {
                     DataType::F16 => crate::cpu::convert_f32_to_f16(&c_f32_vec, bytemuck::cast_slice_mut(res_bytes)),
                     DataType::BF16 => crate::cpu::convert_f32_to_bf16(&c_f32_vec, bytemuck::cast_slice_mut(res_bytes)),
                     DataType::Int8 | DataType::BitNet2 | DataType::BitNet1_6 | DataType::I2_S => {
                          let dst = bytemuck::cast_slice_mut::<u8, i8>(res_bytes);
                          for i in 0..c_f32_vec.len() { dst[i] = c_f32_vec[i] as i8; }
                     }
                     _ => {}
                 }
                 // c_f32_vec drop returns to pool
                 super::pool::TENSOR_POOL.with(|pool| {
                     let mut v = std::mem::take(&mut c_f32_vec);
                     let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                     std::mem::forget(v);
                     pool.borrow_mut().free(unsafe { Vec::from_raw_parts(ptr as *mut u8, len * 4, cap * 4) });
                 });
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
                    let total_elems = (m * n) as usize;
                    let mut c_f32_vec = super::pool::TENSOR_POOL.with(|pool| {
                        let mut p = pool.borrow_mut();
                        let mut buf = p.alloc(total_elems * 4);
                        let (ptr, _len, cap) = (buf.as_mut_ptr(), buf.len(), buf.capacity());
                        std::mem::forget(buf);
                        unsafe { Vec::from_raw_parts(ptr as *mut f32, total_elems, cap / 4) }
                    });

                    crate::cpu::matmul_f32(m as usize, k as usize, n as usize, &a_f32, &b_f32, &mut c_f32_vec);
                    let res_dtype = res.dtype; let (res_bytes, _) = res.get_slice_raw_mut_bytes();
                    match res_dtype {
                        DataType::BF16 => crate::cpu::convert_f32_to_bf16(&c_f32_vec, bytemuck::cast_slice_mut(res_bytes)),
                        DataType::Int8 | DataType::BitNet2 | DataType::BitNet1_6 | DataType::I2_S => {
                             let dst = bytemuck::cast_slice_mut::<u8, i8>(res_bytes);
                             for i in 0..c_f32_vec.len() { dst[i] = c_f32_vec[i] as i8; }
                        }
                        _ => {}
                    }
                    super::pool::TENSOR_POOL.with(|pool| {
                        let mut v = std::mem::take(&mut c_f32_vec);
                        let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                        std::mem::forget(v);
                        pool.borrow_mut().free(unsafe { Vec::from_raw_parts(ptr as *mut u8, len * 4, cap * 4) });
                    });
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
    pub fn subln(&self, normalized_shape: Vec<usize>, weight: Option<&Tensor>, eps: f32) -> PyResult<Tensor> {
        let mut d = 1;
        for dim in &normalized_shape { d *= dim; }
        let total = self.shape.iter().product::<usize>();
        if total % d != 0 {
            return Err(PyValueError::new_err("Invalid normalized_shape for subln"));
        }
        let n = total / d;
        let mut res = Tensor::new_zeros(self.shape.clone(), self.dtype, &self.device)?;

        if self.device == "cpu" {
             match self.dtype {
                 DataType::F32 => {
                     let (x_raw, _) = self.get_slice_raw_f32();
                     let w_raw = if let Some(w) = weight { Some(w.get_slice_raw_f32().0) } else { None };
                     let (out_raw, _) = res.get_slice_raw_mut_f32();
                     crate::cpu::sub_layer_norm_f32(x_raw, w_raw, eps, &normalized_shape, out_raw)?;
                 },
                 _ => return Err(PyValueError::new_err(format!("Unsupported dtype {:?} for subln on CPU", self.dtype))),
             }
        } else {
             return Err(PyValueError::new_err("subln not implemented yet for GPU"));
        }
        Ok(res)
    }

    pub fn rms_norm(&self, normalized_shape: Vec<usize>, weight: Option<&Tensor>, eps: f32) -> PyResult<Tensor> {
        let mut d = 1;
        for dim in &normalized_shape { d *= dim; }
        let total = self.shape.iter().product::<usize>();
        if total % d != 0 {
            println!("[Tensor] RMSNorm FAIL: shape={:?}, norm_shape={:?}, total={}, d={}", self.shape, normalized_shape, total, d);
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
        if weight.dtype != DataType::BitNet2 && weight.dtype != DataType::BitNet1_6 && weight.dtype != DataType::I2_S { 
            return Err(PyValueError::new_err("BitLinear weights must be BitNet2, BitNet1_6, or I2_S")); 
        }
        if weight.shape[1] != k { return Err(PyValueError::new_err("BitLinear shape mismatch (K dimension)")); }
        
        // Result is dequantized to F32 (or could be F16/Int8 target)
        let mut res = Tensor::new_zeros(vec![m, n], DataType::F32, &input.device)?;
        
        if input.device == "cpu" {
            let (a, _) = input.get_slice_raw_i8();
            let (b, _) = weight.get_slice_raw_bitnet();
            let (s, _) = scale.get_slice_raw_f32();
            let (c, _) = res.get_slice_raw_mut_f32();
            
            // Dispatch to the multi-tier BitNet kernel
            crate::cpu::bit_linear_f32(m, k, n, a, b, s, c, weight.dtype);
            
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
            
            crate::backend::execute_bit_linear_into(a_raw, b_raw, s_raw, bias_raw, out_raw, m as u32, k as u32, n as u32, weight.dtype);
        }
        
        Ok(res)
    }

    /// Like execute_bit_linear but with explicit logical N (for weights stored as [N/4, K]).
    pub fn execute_bit_linear_with_n(input: &Tensor, weight: &Tensor, logical_n: usize, scale: &Tensor, bias: Option<&Tensor>) -> PyResult<Tensor> {
        // Input can be [M, K] or [K] for single token
        let k = *input.shape.last().unwrap();
        let m: usize = input.shape[..input.shape.len().saturating_sub(1)].iter().product::<usize>().max(1);
        
        if input.dtype != DataType::Int8 { return Err(PyValueError::new_err("BitLinear input must be Int8")); }
        if weight.dtype != DataType::BitNet2 && weight.dtype != DataType::BitNet1_6 && weight.dtype != DataType::I2_S {
            return Err(PyValueError::new_err("BitLinear weights must be BitNet2, BitNet1_6, or I2_S"));
        }
        
        // Output shape mirrors input shape but last dim becomes logical_n
        let mut out_shape = if input.shape.len() > 1 { input.shape[..input.shape.len()-1].to_vec() } else { vec![1] };
        out_shape.push(logical_n);
        let mut res = Tensor::new_zeros(out_shape, DataType::F32, &input.device)?;
        
        if input.device == "cpu" {
            let (a, _) = input.get_slice_raw_i8();
            let (b, _) = weight.get_slice_raw_bitnet();
            let (s, _) = scale.get_slice_raw_f32();
            let (c, _) = res.get_slice_raw_mut_f32();
            crate::cpu::bit_linear_f32(m, k, logical_n, a, b, s, c, weight.dtype);
            if let Some(b_t) = bias {
                let (bias_v, _) = b_t.get_slice_raw_f32();
                use rayon::prelude::*;
                c.par_chunks_mut(logical_n).for_each(|row| {
                    for j in 0..logical_n { row[j] += bias_v[j]; }
                });
            }
        } else {
            let (a_raw, _) = input.get_slice_raw_bytes();
            let (b_raw, _) = weight.get_slice_raw_bytes();
            let (s_raw, _) = scale.get_slice_raw_bytes();
            let bias_raw = bias.map(|b| b.get_slice_raw_bytes().0).unwrap_or(&[]);
            let (out_raw, _) = res.get_slice_raw_mut_bytes();
            crate::backend::execute_bit_linear_into(a_raw, b_raw, s_raw, bias_raw, out_raw, m as u32, k as u32, logical_n as u32, weight.dtype);
        }
        
        Ok(res)
    }

    pub fn execute_transpose(&self) -> PyResult<Tensor> {
        if self.shape.len() != 2 { return Err(PyValueError::new_err("2D required for transpose")); }
        let new_shape = vec![self.shape[1], self.shape[0]];
        let strides = Self::calculate_default_strides(new_shape.clone());
        Ok(Tensor {
            shape: new_shape,
            strides,
            offset: self.offset,
            device: self.device.clone(),
            dtype: self.dtype,
            storage: self.storage.clone(),
            is_transposed: self.is_transposed,
            name: format!("{}_reshaped", self.name),
            mmap_data: self.mmap_data.clone(),
        })
    }

    pub fn execute_cat(tensors: &[&Self], dim: usize) -> PyResult<Self> {
        if tensors.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("cat: tensors must not be empty"));
        }
        
        let first = tensors[0];
        let dtype = first.dtype;
        let device = &first.device;
        
        for t in tensors {
            if t.dtype != dtype {
                return Err(pyo3::exceptions::PyValueError::new_err("cat: all tensors must have the same dtype"));
            }
            if t.device != *device {
                return Err(pyo3::exceptions::PyValueError::new_err("cat: all tensors must be on the same device"));
            }
            if t.shape.len() != first.shape.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("cat: all tensors must have the same number of dimensions"));
            }
            for i in 0..first.shape.len() {
                if i != dim && t.shape[i] != first.shape[i] {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!("cat: dimension mismatch at dim {}", i)));
                }
            }
        }
        
        // Allocate the result buffer ONCE in RAM (raw bytes)
        let bytes_per_elem = dtype.size();
        let mut out_shape = first.shape.clone();
        let total_dim: usize = tensors.iter().map(|t| t.shape[dim]).sum();
        out_shape[dim] = total_dim;
        let total_out_elems: usize = out_shape.iter().product();
        let total_out_bytes = total_out_elems * bytes_per_elem;

        let mut raw_out = unsafe {
            let cap = (total_out_bytes + 7) / 8;
            let v_f64 = vec![0.0f64; cap];
            let ptr = v_f64.as_ptr();
            std::mem::forget(v_f64);
            Vec::from_raw_parts(ptr as *mut u8, total_out_bytes, cap * 8)
        };

        // For each input tensor, copy its bytes into the correct window of raw_out.
        // SSD tensors use MSTS load_to_buffer (ring-buffered, no intermediate Vec).
        // RAM/Vulkan tensors use a direct memcpy.
        //
        // For dim=0 (and contiguous row-major layout) every tensor contributes a contiguous
        // block. We compute the destination byte offset per tensor.
        //
        // For dim>0 we fall back to the typed cat kernels (RAM tensors only).
        let all_contiguous_dim0 = dim == 0;

        if all_contiguous_dim0 {
            let mut dest_byte_offset: u64 = 0;
            for t in tensors.iter() {
                let t_bytes = (t.shape.iter().product::<usize>()) * bytes_per_elem;
                if t.is_ssd() {
                    // True MSTS: stream tile-by-tile directly into raw_out window
                    t.load_to_buffer(&mut raw_out, dest_byte_offset)?;
                } else if t.device == "vulkan" {
                    let (slice, _) = t.get_slice_raw_bytes();
                    let dst_start = dest_byte_offset as usize;
                    raw_out[dst_start..dst_start + t_bytes].copy_from_slice(slice);
                } else {
                    let (slice, _) = t.get_slice_raw_bytes();
                    let dst_start = dest_byte_offset as usize;
                    raw_out[dst_start..dst_start + t_bytes].copy_from_slice(slice);
                }
                dest_byte_offset += t_bytes as u64;
            }
        } else {
            // dim > 0: need typed strides logic – load everything to RAM first, use typed kernels
            let mut cpu_tensors_owned = Vec::new();
            for t in tensors.iter() {
                if t.is_ssd() {
                    let storage = t.execute_load_to_storage_cpu()?;
                    let strides = Self::calculate_default_strides(t.shape.clone());
                    cpu_tensors_owned.push(Tensor {
                        shape: t.shape.clone(), strides, offset: 0,
                        device: "cpu".to_string(), dtype: t.dtype, storage,
                        is_transposed: false,
                        name: format!("{}_hssd", t.name), mmap_data: None,
                    });
                } else if t.device == "vulkan" {
                    let (slice, _) = t.get_slice_raw_bytes();
                    let storage = t.raw_to_storage(slice);
                    let strides = Self::calculate_default_strides(t.shape.clone());
                    cpu_tensors_owned.push(Tensor {
                        shape: t.shape.clone(), strides, offset: 0,
                        device: "cpu".to_string(), dtype: t.dtype, storage,
                        is_transposed: false,
                        name: format!("{}_hv", t.name), mmap_data: None,
                    });
                } else {
                    cpu_tensors_owned.push((*t).clone());
                }
            }
            let cpu_tensors: Vec<&Tensor> = cpu_tensors_owned.iter().collect();
            raw_out = match dtype {
                DataType::F32  => unsafe {
                    let v = crate::cpu::cat_f32(&cpu_tensors, dim);
                    let ptr = v.as_ptr();
                    let len = v.len() * 4;
                    let cap = v.capacity() * 4;
                    std::mem::forget(v);
                    Vec::from_raw_parts(ptr as *mut u8, len, cap)
                },
                DataType::F16  => unsafe {
                    let v = crate::cpu::cat_f16(&cpu_tensors, dim);
                    let ptr = v.as_ptr();
                    let len = v.len() * 2;
                    let cap = v.capacity() * 2;
                    std::mem::forget(v);
                    Vec::from_raw_parts(ptr as *mut u8, len, cap)
                },
                DataType::BF16 => unsafe {
                    let v = crate::cpu::cat_bf16(&cpu_tensors, dim);
                    let ptr = v.as_ptr();
                    let len = v.len() * 2;
                    let cap = v.capacity() * 2;
                    std::mem::forget(v);
                    Vec::from_raw_parts(ptr as *mut u8, len, cap)
                },
                DataType::Int8 | DataType::BitNet2 | DataType::BitNet1_6 | DataType::I2_S => unsafe {
                    let v = crate::cpu::cat_i8(&cpu_tensors, dim);
                    let ptr = v.as_ptr();
                    let len = v.len();
                    let cap = v.capacity();
                    std::mem::forget(v);
                    Vec::from_raw_parts(ptr as *mut u8, len, cap)
                },
            };
        }

        // Build the typed Storage from raw bytes
        let storage = match dtype {
            DataType::F32  => unsafe {
                let ptr = raw_out.as_ptr();
                let len = raw_out.len() / 4;
                let cap = raw_out.capacity() / 4;
                std::mem::forget(raw_out);
                Storage::F32(Vec::from_raw_parts(ptr as *mut f32, len, cap))
            },
            DataType::F16  => unsafe {
                let ptr = raw_out.as_ptr();
                let len = raw_out.len() / 2;
                let cap = raw_out.capacity() / 2;
                std::mem::forget(raw_out);
                Storage::F16(Vec::from_raw_parts(ptr as *mut half::f16, len, cap))
            },
            DataType::BF16 => unsafe {
                let ptr = raw_out.as_ptr();
                let len = raw_out.len() / 2;
                let cap = raw_out.capacity() / 2;
                std::mem::forget(raw_out);
                Storage::BF16(Vec::from_raw_parts(ptr as *mut half::bf16, len, cap))
            },
            DataType::Int8 => unsafe {
                let ptr = raw_out.as_ptr();
                let len = raw_out.len();
                let cap = raw_out.capacity();
                std::mem::forget(raw_out);
                Storage::Int8(Vec::from_raw_parts(ptr as *mut i8, len, cap))
            },
            DataType::BitNet2 | DataType::BitNet1_6 => unsafe {
                let ptr = raw_out.as_ptr();
                let len = raw_out.len();
                let cap = raw_out.capacity();
                std::mem::forget(raw_out);
                Storage::BitNet(Vec::from_raw_parts(ptr as *mut u8, len, cap))
            },
            DataType::I2_S => unsafe {
                let ptr = raw_out.as_ptr();
                let len = raw_out.len();
                let cap = raw_out.capacity();
                std::mem::forget(raw_out);
                Storage::I2_S(Vec::from_raw_parts(ptr as *mut u8, len, cap))
            },
        };

        let strides = Self::calculate_default_strides(out_shape.clone());
        let mut out = Tensor {
            shape: out_shape,
            strides,
            offset: 0,
            device: "cpu".to_string(),
            dtype,
            storage,
            is_transposed: false,
            name: format!("{}_cat", first.name),
            mmap_data: None,
        };

        if device == "vulkan" {
            out = out.to_device(device)?;
        }

        Ok(out)
    }

    pub fn execute_stack(tensors: &[&Self], dim: usize) -> PyResult<Self> {
        let mut unsqueezed = Vec::new();
        for t in tensors {
            unsqueezed.push(t.execute_unsqueeze(dim)?);
        }
        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        Self::execute_cat(&refs, dim)
    }

    pub fn execute_split(&self, split_size: usize, dim: usize) -> PyResult<Vec<Tensor>> {
        if dim >= self.shape.len() { return Err(PyValueError::new_err("Dim out of range")); }
        let dim_size = self.shape[dim];
        let mut results = Vec::new();
        let mut current = 0;
        
        while current < dim_size {
            let size = std::cmp::min(split_size, dim_size - current);
            let mut new_shape = self.shape.clone();
            new_shape[dim] = size;
            
            let offset = self.offset + current * self.strides[dim];
            
            results.push(Tensor {
                shape: new_shape,
                strides: self.strides.clone(),
                offset,
                device: self.device.clone(),
                dtype: self.dtype,
                storage: self.storage.clone(),
                is_transposed: self.is_transposed,
                name: format!("{}_split_{}", self.name, results.len()),
                mmap_data: self.mmap_data.clone(),
            });
            
            current += size;
        }
        Ok(results)
    }

    pub fn execute_chunk(&self, chunks: usize, dim: usize) -> PyResult<Vec<Tensor>> {
        if dim >= self.shape.len() { return Err(PyValueError::new_err("Dim out of range")); }
        let dim_size = self.shape[dim];
        if chunks == 0 { return Err(PyValueError::new_err("chunks must be > 0")); }
        let split_size = (dim_size + chunks - 1) / chunks;
        self.execute_split(split_size, dim)
    }
}
