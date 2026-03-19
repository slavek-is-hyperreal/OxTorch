use pyo3::prelude::*;
use rayon::prelude::*;
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
            let mut out_t = Tensor::new_zeros(self.shape.clone(), self.dtype, "cpu")?;
            let (in_raw, _) = self.get_slice_raw_bytes();
            let (out_raw, _) = out_t.get_slice_raw_mut_bytes();
            
            let stride = self.shape[d_usize..].iter().product::<usize>();
            let outer = self.shape[..d_usize].iter().product::<usize>();
            let dim_size = self.shape[d_usize];
            let inner = stride / dim_size;

            let compute_row = |i: usize, k: usize, in_ptr: *const u8, out_ptr: *mut u8| {
                let row_offset = (i * stride + k) * self.dtype.size();
                match self.dtype {
                    DataType::F32 => {
                        let mut row = vec![0.0f32; dim_size];
                        for j in 0..dim_size { row[j] = unsafe { *(in_ptr.add(row_offset + j * inner * 4) as *const f32) }; }
                        crate::cpu::softmax_f32_dispatch(&mut row, is_log);
                        for j in 0..dim_size { unsafe { *(out_ptr.add(row_offset + j * inner * 4) as *mut f32) = row[j]; } }
                    },
                    DataType::F16 => {
                        let mut row = vec![half::f16::ZERO; dim_size];
                        for j in 0..dim_size { row[j] = unsafe { *(in_ptr.add(row_offset + j * inner * 2) as *const half::f16) }; }
                        crate::cpu::softmax_f16_dispatch(&mut row, is_log);
                        for j in 0..dim_size { unsafe { *(out_ptr.add(row_offset + j * inner * 2) as *mut half::f16) = row[j]; } }
                    },
                    DataType::BF16 => {
                        let mut row = vec![half::bf16::ZERO; dim_size];
                        for j in 0..dim_size { row[j] = unsafe { *(in_ptr.add(row_offset + j * inner * 2) as *const half::bf16) }; }
                        crate::cpu::softmax_bf16_dispatch(&mut row, is_log);
                        for j in 0..dim_size { unsafe { *(out_ptr.add(row_offset + j * inner * 2) as *mut half::bf16) = row[j]; } }
                    },
                    DataType::Int8 => {
                        let mut row = vec![0i8; dim_size];
                        for j in 0..dim_size { row[j] = unsafe { *(in_ptr.add(row_offset + j * inner) as *const i8) }; }
                        crate::cpu::softmax_i8_dispatch(&mut row, is_log);
                        for j in 0..dim_size { unsafe { *(out_ptr.add(row_offset + j * inner) as *mut i8) = row[j]; } }
                    },
                }
            };

            const CPU_PARALLEL_THRESHOLD: usize = 32768;
            if self.shape.iter().product::<usize>() > CPU_PARALLEL_THRESHOLD {
                (0..outer).into_par_iter().for_each(|i| {
                    let in_p = in_raw.as_ptr();
                    let out_p = out_raw.as_ptr() as *mut u8;
                    for k in 0..inner { compute_row(i, k, in_p, out_p); }
                });
            } else {
                let in_p = in_raw.as_ptr();
                let out_p = out_raw.as_ptr() as *mut u8;
                for i in 0..outer {
                    for k in 0..inner { compute_row(i, k, in_p, out_p); }
                }
            }
            Ok(out_t)
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
        let d_usize = if let Some(d) = dim {
            let idx = if d < 0 { (self.shape.len() as i64 + d) as usize } else { d as usize };
            out_shape[idx] = 1;
            Some(idx)
        } else {
            out_shape = vec![1];
            None
        };

        let (in_raw, _) = self.get_slice_raw_bytes();
        let mut out_t = Tensor::new_zeros(out_shape, self.dtype, "cpu")?;
        let (out_raw, _) = out_t.get_slice_raw_mut_bytes();

        match d_usize {
            None => {
                let val: f32 = match self.dtype {
                    DataType::F32 => {
                        let slice = bytemuck::cast_slice::<u8, f32>(in_raw);
                        match op {
                            "sum" => crate::cpu::sum_f32_dispatch(slice),
                            "mean" => crate::cpu::sum_f32_dispatch(slice) / (slice.len() as f32),
                            "max" => slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                            _ => 0.0,
                        }
                    },
                    DataType::F16 => {
                        let slice = bytemuck::cast_slice::<u8, half::f16>(in_raw);
                        match op {
                            "sum" => crate::cpu::sum_f16_dispatch(slice),
                            "max" => crate::cpu::max_f16_dispatch(slice, f32::NEG_INFINITY),
                            _ => 0.0,
                        }
                    },
                    DataType::BF16 => {
                        let slice = bytemuck::cast_slice::<u8, half::bf16>(in_raw);
                        match op {
                            "sum" => crate::cpu::sum_bf16_dispatch(slice),
                            "max" => crate::cpu::max_bf16_dispatch(slice, f32::NEG_INFINITY),
                            _ => 0.0,
                        }
                    },
                    DataType::Int8 => {
                        let slice = bytemuck::cast_slice::<u8, i8>(in_raw);
                        match op {
                            "sum" => crate::cpu::sum_i8_dispatch(slice) as f32,
                            "max" => crate::cpu::max_i8_dispatch(slice, i8::MIN) as f32,
                            _ => 0.0,
                        }
                    },
                };
                match self.dtype {
                    DataType::F32 => unsafe { *(out_raw.as_ptr() as *mut f32) = val; },
                    DataType::F16 => unsafe { *(out_raw.as_ptr() as *mut half::f16) = half::f16::from_f32(val); },
                    DataType::BF16 => unsafe { *(out_raw.as_ptr() as *mut half::bf16) = half::bf16::from_f32(val); },
                    DataType::Int8 => unsafe { *(out_raw.as_ptr() as *mut i8) = val as i8; },
                }
            },
            Some(d) => {
                let stride = self.shape[d..].iter().product::<usize>();
                let outer = self.shape[..d].iter().product::<usize>();
                let dim_size = self.shape[d];
                let inner = stride / dim_size;

                for i in 0..outer {
                    for k in 0..inner {
                        let row_base = i * stride + k;
                        let mut row = Vec::with_capacity(dim_size);
                        for j in 0..dim_size {
                            match self.dtype {
                                DataType::F32 => row.push(unsafe { *(in_raw.as_ptr().add((row_base + j * inner) * 4) as *const f32) }),
                                DataType::F16 => row.push(unsafe { (*(in_raw.as_ptr().add((row_base + j * inner) * 2) as *const half::f16)).to_f32() }),
                                DataType::BF16 => row.push(unsafe { (*(in_raw.as_ptr().add((row_base + j * inner) * 2) as *const half::bf16)).to_f32() }),
                                DataType::Int8 => row.push(unsafe { *(in_raw.as_ptr().add(row_base + j * inner) as *const i8) as f32 }),
                            }
                        }
                        let acc = match op {
                            "sum" => row.iter().sum::<f32>(),
                            "max" => row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                            "mean" => row.iter().sum::<f32>() / (dim_size as f32),
                            _ => 0.0,
                        };
                        match self.dtype {
                            DataType::F32 => unsafe { *(out_raw.as_ptr().add((i * inner + k) * 4) as *mut f32) = acc; },
                            DataType::F16 => unsafe { *(out_raw.as_ptr().add((i * inner + k) * 2) as *mut half::f16) = half::f16::from_f32(acc); },
                            DataType::BF16 => unsafe { *(out_raw.as_ptr().add((i * inner + k) * 2) as *mut half::bf16) = half::bf16::from_f32(acc); },
                            DataType::Int8 => unsafe { *(out_raw.as_ptr().add(i * inner + k) as *mut i8) = acc as i8; },
                        }
                    }
                }
            }
        }
        Ok(out_t)
    }
}
