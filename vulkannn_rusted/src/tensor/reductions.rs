use pyo3::prelude::*;
use super::{Tensor, DataType};

impl Tensor {
    pub fn execute_softmax(&self, dim: i64, is_log: bool) -> PyResult<Tensor> {
        let d_usize = if dim < 0 { (self.shape.len() as i64 + dim) as usize } else { dim as usize };
        if d_usize >= self.shape.len() { return Err(pyo3::exceptions::PyValueError::new_err("Invalid dimension")); }
        if self.device != "cpu" {
             let (a_raw, _) = self.get_slice_raw_bytes();
             let mut out_t = Tensor::new_zeros(self.shape.clone(), self.dtype, &self.device)?;
             let (out_raw, _) = out_t.get_slice_raw_mut_bytes();
             let stride = self.shape[d_usize..].iter().product::<usize>();
             let outer = self.shape[..d_usize].iter().product::<usize>();
             let dim_size = self.shape[d_usize];
             let inner = stride / dim_size;
             crate::backend::execute_softmax_into(a_raw, out_raw, dim_size as u32, (outer * inner) as u32, is_log, self.dtype);
             Ok(out_t)
        } else {
            let mut out = self.clone();
            let (v, _) = out.get_slice_raw_mut_f32();
            let stride = self.shape[d_usize..].iter().product::<usize>();
            let outer = self.shape[..d_usize].iter().product::<usize>();
            let dim_size = self.shape[d_usize];
            let inner = stride / dim_size;
            for i in 0..outer {
                for k in 0..inner {
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..dim_size { max_val = max_val.max(v[i * stride + j * inner + k]); }
                    let mut sum = 0.0;
                    for j in 0..dim_size {
                        let val = (v[i * stride + j * inner + k] - max_val).exp();
                        v[i * stride + j * inner + k] = val;
                        sum += val;
                    }
                    if is_log {
                        let log_sum = sum.ln();
                        for j in 0..dim_size { v[i * stride + j * inner + k] = (v[i * stride + j * inner + k]).ln() - log_sum; }
                    } else {
                        for j in 0..dim_size { v[i * stride + j * inner + k] /= sum; }
                    }
                }
            }
            Ok(out)
        }
    }

    pub fn execute_reduce(&self, op: &str, dim: Option<i64>) -> PyResult<Tensor> {
        if self.device != "cpu" && dim.is_none() {
            let (a_raw, _) = self.get_slice_raw_bytes();
            let blocks = crate::backend::execute_reduce(a_raw, op, self.dtype);
            let val = match op {
                "sum" => blocks.iter().sum::<f32>(),
                "mean" => blocks.iter().sum::<f32>() / (self.shape.iter().product::<usize>() as f32),
                _ => blocks[0], 
            };
            return Ok(Tensor::new_from_vec(vec![val], vec![1], self.dtype, "cpu", "reduced")?);
        }
        let mut out_shape = self.shape.clone();
        if let Some(d) = dim {
            let d_usize = if d < 0 { (self.shape.len() as i64 + d) as usize } else { d as usize };
            out_shape[d_usize] = 1;
        } else {
            out_shape = vec![1];
        }

        let mut res = Tensor::new_zeros(out_shape, DataType::F32, "cpu")?;
        let (v_in, _) = self.get_slice_raw_f32();
        let (v_out, _) = res.get_slice_raw_mut_f32();

        if dim.is_none() {
            let val = match op {
                "sum" => (unsafe { crate::cpu::sum_f32_dispatch(v_in) }),
                "mean" => (unsafe { crate::cpu::sum_f32_dispatch(v_in) }) / (v_in.len() as f32),
                "max" => v_in.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                "min" => v_in.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                _ => 0.0,
            };
            v_out[0] = val;
        }
        Ok(res)
    }
}
