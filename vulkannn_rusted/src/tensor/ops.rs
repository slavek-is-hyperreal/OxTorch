use pyo3::prelude::*;
use super::{Tensor, DataType};

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
            match self.dtype {
                DataType::F32 => {
                    let (a, _) = self.get_slice_raw_f32();
                    let (b, _) = other.get_slice_raw_f32();
                    let (res, _) = out.get_slice_raw_mut_f32();
                    match op {
                        "add" => crate::cpu::elementwise_add_f32(a, b, res),
                        "sub" => crate::cpu::elementwise_sub_f32(a, b, res),
                        "mul" => crate::cpu::elementwise_mul_f32(a, b, res),
                        "div" => crate::cpu::elementwise_div_f32(a, b, res),
                        _ => {},
                    }
                },
                DataType::F16 => {
                    let (a, _) = self.get_slice_raw_f16();
                    let (b, _) = other.get_slice_raw_f16();
                    let (res, _) = out.get_slice_raw_mut_f16();
                    crate::cpu::elementwise_op_f16(a, b, res, op);
                },
                DataType::BF16 => {
                    let (a, _) = self.get_slice_raw_bf16();
                    let (b, _) = other.get_slice_raw_bf16();
                    let (res, _) = out.get_slice_raw_mut_bf16();
                    crate::cpu::elementwise_op_bf16(a, b, res, op);
                },
                DataType::Int8 => {
                    let (a, _) = self.get_slice_raw_i8();
                    let (b, _) = other.get_slice_raw_i8();
                    let (res, _) = out.get_slice_raw_mut_i8();
                    crate::cpu::elementwise_op_i8(a, b, res, op);
                },
                DataType::Ternary => { return Err(pyo3::exceptions::PyValueError::new_err("Ops not supported for Ternary")); }
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
            match self.dtype {
                DataType::F32 => {
                    let (v_in, _) = self.get_slice_raw_f32();
                    let (v_out, _) = out.get_slice_raw_mut_f32();
                    match op {
                        "relu" => crate::cpu::relu_f32(v_in, v_out),
                        "gelu" => { v_out.copy_from_slice(v_in); crate::cpu::gelu_f32_inplace(v_out); },
                        _ => { v_out.copy_from_slice(v_in); },
                    }
                },
                DataType::F16 => {
                    let (v_in, _) = self.get_slice_raw_f16();
                    let (v_out, _) = out.get_slice_raw_mut_f16();
                    match op {
                        "relu" => crate::cpu::relu_f16(v_in, v_out),
                        "gelu" => { v_out.copy_from_slice(v_in); crate::cpu::gelu_f16_inplace(v_out); },
                        _ => { v_out.copy_from_slice(v_in); },
                    }
                },
                DataType::BF16 => {
                    let (v_in, _) = self.get_slice_raw_bf16();
                    let (v_out, _) = out.get_slice_raw_mut_bf16();
                    match op {
                        "relu" => crate::cpu::relu_bf16(v_in, v_out),
                        "gelu" => { v_out.copy_from_slice(v_in); crate::cpu::gelu_bf16_inplace(v_out); },
                        _ => { v_out.copy_from_slice(v_in); },
                    }
                },
                DataType::Int8 => {
                    let (v_in, _) = self.get_slice_raw_i8();
                    let (v_out, _) = out.get_slice_raw_mut_i8();
                    v_out.copy_from_slice(v_in);
                    match op {
                        "relu" => crate::cpu::relu_i8_inplace(v_out),
                        "gelu" => crate::cpu::gelu_i8_dispatch(v_out),
                        _ => {},
                    }
                },
                DataType::Ternary => { return Err(pyo3::exceptions::PyValueError::new_err("Ops not supported for Ternary")); }
            }
        } else {
            let (input_raw, _) = self.get_slice_raw_bytes();
            let (out_raw, _) = out.get_slice_raw_mut_bytes();
            crate::backend::execute_activation_into(input_raw, op, param1, param2, out_raw, self.dtype, self.device == "hybrid", false);
        }
        Ok(out)
    }

    pub fn unary_op_into(&self, target: &mut Tensor, op: &str, param1: f32, param2: f32) -> PyResult<()> {
        if self.shape != target.shape { return Err(pyo3::exceptions::PyValueError::new_err("Shapes must match")); }
        
        if self.device == "cpu" {
             match self.dtype {
                DataType::F32 => {
                    let (v_in, _) = self.get_slice_raw_f32();
                    let (v_out, _) = target.get_slice_raw_mut_f32();
                    match op {
                        "relu" => crate::cpu::relu_f32(v_in, v_out),
                        "gelu" => { v_out.copy_from_slice(v_in); crate::cpu::gelu_f32_inplace(v_out); },
                        _ => { v_out.copy_from_slice(v_in); },
                    }
                },
                DataType::F16 => {
                    let (v_in, _) = self.get_slice_raw_f16();
                    let (v_out, _) = target.get_slice_raw_mut_f16();
                    match op {
                        "relu" => crate::cpu::relu_f16(v_in, v_out),
                        "gelu" => { v_out.copy_from_slice(v_in); crate::cpu::gelu_f16_inplace(v_out); },
                        _ => { v_out.copy_from_slice(v_in); },
                    }
                },
                DataType::BF16 => {
                    let (v_in, _) = self.get_slice_raw_bf16();
                    let (v_out, _) = target.get_slice_raw_mut_bf16();
                    match op {
                        "relu" => crate::cpu::relu_bf16(v_in, v_out),
                        "gelu" => { v_out.copy_from_slice(v_in); crate::cpu::gelu_bf16_inplace(v_out); },
                        _ => { v_out.copy_from_slice(v_in); },
                    }
                },
                DataType::Int8 => {
                    let (v_in, _) = self.get_slice_raw_i8();
                    let (v_out, _) = target.get_slice_raw_mut_i8();
                    v_out.copy_from_slice(v_in);
                    match op {
                        "relu" => crate::cpu::relu_i8_swar(v_out),
                        _ => {},
                    }
                },
                DataType::Ternary => { return Err(pyo3::exceptions::PyValueError::new_err("Ops not supported for Ternary")); }
             }
        } else {
             let (input_raw, _) = self.get_slice_raw_bytes();
             let (out_raw, _) = target.get_slice_raw_mut_bytes();
             crate::backend::execute_activation_into(input_raw, op, param1, param2, out_raw, self.dtype, self.device == "hybrid", false);
        }
        Ok(())
    }

    pub fn act_into_raw_parallel_f32(slice: &mut [f32], op: &str, _param1: f32, _param2: f32) {
        match op {
            "relu" => crate::cpu::relu_f32_inplace(slice),
            "gelu" => crate::cpu::gelu_f32_inplace(slice),
            _ => {},
        }
    }
}
