use pyo3::prelude::*;
use numpy::ToPyArray;
use super::{Tensor, DataType, Storage};

impl Tensor {
    pub fn execute_to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        let vec = match self.dtype {
            DataType::F32 => {
                if self.is_ssd() { self.load_to_f32_vec_msts() } else {
                    let (slice, _) = self.get_slice_raw_f32();
                    slice.to_vec()
                }
            },
            DataType::F16 => {
                if self.is_ssd() { self.load_to_f32_vec_msts() } else {
                    let (slice, _) = self.get_slice_raw_f16();
                    let mut vec = vec![0.0; slice.len()];
                    crate::cpu::convert_f16_to_f32(slice, &mut vec);
                    vec
                }
            },
            DataType::BF16 => {
                if self.is_ssd() { self.load_to_f32_vec_msts() } else {
                    let (slice, _) = self.get_slice_raw_bf16();
                    let mut vec = vec![0.0; slice.len()];
                    crate::cpu::convert_bf16_to_f32(slice, &mut vec);
                    vec
                }
            },
            DataType::Int8 => {
                if self.is_ssd() { self.load_to_f32_vec_msts() } else {
                    let (slice, _) = self.get_slice_raw_i8();
                    slice.iter().map(|&x| x as f32).collect()
                }
            },
            DataType::BitNet2 | DataType::BitNet1_6 => {
                if self.is_ssd() { self.load_to_f32_vec_msts() } else {
                    let (slice, _) = self.get_slice_raw_bitnet();
                    // Basic dequantizer for NumPy export (slow but correct)
                    let mut vec = Vec::with_capacity(self.shape.iter().product());
                    if self.dtype == DataType::BitNet2 {
                        for &byte in slice {
                            for shift in (0..8).step_by(2) {
                                let val = (byte >> shift) & 0x03;
                                vec.push((val as i8 - 1) as f32); // 0->-1, 1->0, 2->1
                                if vec.len() == vec.capacity() { break; }
                            }
                            if vec.len() == vec.capacity() { break; }
                        }
                    } else {
                        // BitNet 1.6: 5 trits per byte (Base-3)
                        for &byte in slice {
                            let mut b = byte;
                            for _ in 0..5 {
                                let trit = b % 3;
                                vec.push((trit as i8 - 1) as f32);
                                b /= 3;
                                if vec.len() == vec.capacity() { break; }
                            }
                            if vec.len() == vec.capacity() { break; }
                        }
                    }
                    vec
                }
            }
        };
        let array = numpy::ndarray::Array::from_shape_vec(self.shape.clone(), vec).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(array.to_pyarray_bound(py))
    }

    pub fn to_numpy_f32_vec(&self) -> Vec<f32> {
        if self.is_ssd() { self.load_to_f32_vec_msts() } else {
            match self.dtype {
                DataType::F32 => self.get_slice_raw_f32().0.to_vec(),
                DataType::F16 => {
                    let (s, _) = self.get_slice_raw_f16();
                    let mut v = vec![0.0; s.len()];
                    crate::cpu::convert_f16_to_f32(s, &mut v);
                    v
                },
                DataType::BF16 => {
                    let (s, _) = self.get_slice_raw_bf16();
                    let mut v = vec![0.0; s.len()];
                    crate::cpu::convert_bf16_to_f32(s, &mut v);
                    v
                },
                DataType::Int8 => {
                    let (s, _) = self.get_slice_raw_i8();
                    s.iter().map(|&x| x as f32).collect()
                },
                DataType::BitNet2 | DataType::BitNet1_6 => {
                    let (s, _) = self.get_slice_raw_bitnet();
                    let total = self.shape.iter().product::<usize>();
                    let mut vec = Vec::with_capacity(total);
                    if self.dtype == DataType::BitNet2 {
                        for &byte in s {
                            for shift in (0..8).step_by(2) {
                                let val = (byte >> shift) & 0x03;
                                vec.push((val as i8 - 1) as f32);
                                if vec.len() == total { break; }
                            }
                            if vec.len() == total { break; }
                        }
                    } else {
                        for &byte in s {
                            let mut b = byte;
                            for _ in 0..5 {
                                vec.push(((b % 3) as i8 - 1) as f32);
                                b /= 3;
                                if vec.len() == total { break; }
                            }
                            if vec.len() == total { break; }
                        }
                    }
                    vec
                }
            }
        }
    }

    pub fn get_slice_raw_f32(&self) -> (&[f32], usize) {
        match &self.storage {
            Storage::F32(v) => {
                let size = self.shape.iter().product::<usize>();
                let end = std::cmp::min(self.offset + size, v.len());
                (&v[self.offset..end], end - self.offset)
            },
            _ => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_f32(&mut self) -> (&mut [f32], usize) {
        let offset = self.offset;
        let size = self.shape.iter().product::<usize>();
        match &mut self.storage {
            Storage::F32(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (&mut v[offset..end], end - offset)
            },
            _ => (&mut [], 0),
        }
    }

    pub fn get_slice_raw_f16(&self) -> (&[half::f16], usize) {
        match &self.storage {
            Storage::F16(v) => {
                let size = self.shape.iter().product::<usize>();
                let end = std::cmp::min(self.offset + size, v.len());
                (&v[self.offset..end], end - self.offset)
            },
            _ => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_f16(&mut self) -> (&mut [half::f16], usize) {
        let offset = self.offset;
        let size = self.shape.iter().product::<usize>();
        match &mut self.storage {
            Storage::F16(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (&mut v[offset..end], end - offset)
            },
            _ => (&mut [], 0),
        }
    }

    pub fn get_slice_raw_bf16(&self) -> (&[half::bf16], usize) {
        match &self.storage {
            Storage::BF16(v) => {
                let size = self.shape.iter().product::<usize>();
                let end = std::cmp::min(self.offset + size, v.len());
                (&v[self.offset..end], end - self.offset)
            },
            _ => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_bf16(&mut self) -> (&mut [half::bf16], usize) {
        let offset = self.offset;
        let size = self.shape.iter().product::<usize>();
        match &mut self.storage {
            Storage::BF16(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (&mut v[offset..end], end - offset)
            },
            _ => (&mut [], 0),
        }
    }

    pub fn get_slice_raw_i8(&self) -> (&[i8], usize) {
        match &self.storage {
            Storage::Int8(v) => {
                let size = self.shape.iter().product::<usize>();
                let end = std::cmp::min(self.offset + size, v.len());
                (&v[self.offset..end], end - self.offset)
            },
            _ => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_i8(&mut self) -> (&mut [i8], usize) {
        let offset = self.offset;
        let size = self.shape.iter().product::<usize>();
        match &mut self.storage {
            Storage::Int8(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (&mut v[offset..end], end - offset)
            },
            _ => (&mut [], 0),
        }
    }

    pub fn get_slice_raw_bitnet(&self) -> (&[u8], usize) {
        match &self.storage {
            Storage::BitNet(v) => {
                // For packed types, offset is in bytes.
                // We assume offset is already correctly calculated for packing.
                let size_bytes = match self.dtype {
                    DataType::BitNet2 => (self.shape.iter().product::<usize>() + 3) / 4,
                    DataType::BitNet1_6 => (self.shape.iter().product::<usize>() + 4) / 5,
                    _ => 0,
                };
                let end = std::cmp::min(self.offset + size_bytes, v.len());
                (&v[self.offset..end], end - self.offset)
            },
            _ => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_bitnet(&mut self) -> (&mut [u8], usize) {
        let offset = self.offset;
        let size_bytes = match self.dtype {
            DataType::BitNet2 => (self.shape.iter().product::<usize>() + 3) / 4,
            DataType::BitNet1_6 => (self.shape.iter().product::<usize>() + 4) / 5,
            _ => 0,
        };
        match &mut self.storage {
            Storage::BitNet(v) => {
                let end = std::cmp::min(offset + size_bytes, v.len());
                (&mut v[offset..end], end - offset)
            },
            _ => (&mut [], 0),
        }
    }

    pub fn get_slice_raw_bytes(&self) -> (&[u8], usize) {
        let offset = self.offset;
        let size = self.shape.iter().product::<usize>();
        let _bpe = match self.dtype {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::Int8 => 1,
            DataType::BitNet2 | DataType::BitNet1_6 => 0,
        };
        match &self.storage {
            Storage::F32(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (bytemuck::cast_slice(&v[offset..end]), (end - offset) * 4)
            },
            Storage::F16(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (bytemuck::cast_slice(&v[offset..end]), (end - offset) * 2)
            },
            Storage::BF16(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (bytemuck::cast_slice(&v[offset..end]), (end - offset) * 2)
            },
            Storage::Int8(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (bytemuck::cast_slice(&v[offset..end]), end - offset)
            },
            Storage::BitNet(v) => {
                let size_bytes = if self.dtype == DataType::BitNet2 { (size + 3) / 4 } else { (size + 4) / 5 };
                let end = std::cmp::min(offset + size_bytes, v.len());
                (&v[offset..end], end - offset)
            },
            Storage::None => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_bytes(&mut self) -> (&mut [u8], usize) {
        let offset = self.offset;
        let size = self.shape.iter().product::<usize>();
        match &mut self.storage {
            Storage::F32(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (bytemuck::cast_slice_mut(&mut v[offset..end]), (end - offset) * 4)
            },
            Storage::F16(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (bytemuck::cast_slice_mut(&mut v[offset..end]), (end - offset) * 2)
            },
            Storage::BF16(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (bytemuck::cast_slice_mut(&mut v[offset..end]), (end - offset) * 2)
            },
            Storage::Int8(v) => {
                let end = std::cmp::min(offset + size, v.len());
                (bytemuck::cast_slice_mut(&mut v[offset..end]), end - offset)
            },
            Storage::BitNet(v) => {
                let size_bytes = if self.dtype == DataType::BitNet2 { (size + 3) / 4 } else { (size + 4) / 5 };
                let end = std::cmp::min(offset + size_bytes, v.len());
                (&mut v[offset..end], end - offset)
            },
            Storage::None => (&mut [], 0),
        }
    }

    pub fn raw_to_storage(&self, raw: &[u8]) -> Storage {
        match self.dtype {
            DataType::F32 => unsafe {
                let v_f32 = std::slice::from_raw_parts(raw.as_ptr() as *const f32, raw.len() / 4);
                Storage::F32(v_f32.to_vec())
            },
            DataType::F16 => unsafe {
                let v_f16 = std::slice::from_raw_parts(raw.as_ptr() as *const half::f16, raw.len() / 2);
                Storage::F16(v_f16.to_vec())
            },
            DataType::BF16 => unsafe {
                let v_bf16 = std::slice::from_raw_parts(raw.as_ptr() as *const half::bf16, raw.len() / 2);
                Storage::BF16(v_bf16.to_vec())
            },
            DataType::Int8 => unsafe {
                let v_i8 = std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len());
                Storage::Int8(v_i8.to_vec())
            },
            DataType::BitNet2 | DataType::BitNet1_6 => {
                Storage::BitNet(raw.to_vec())
            },
        }
    }

    pub fn execute_index_select(&self, dim: usize, indices: &Tensor) -> PyResult<Tensor> {
        if dim != 0 {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err("OxTorch native SIMD currently only supports index_select on dim=0"));
        }
        
        // Assert indices is at least 1D
        if indices.shape.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Indices tensor cannot be a scalar"));
        }
        
        let feature_len = self.shape[1..].iter().product::<usize>();
        let num_indices = indices.shape.iter().product::<usize>();
        
        let mut out_shape = indices.shape.clone();
        out_shape.extend_from_slice(&self.shape[1..]);

        let mut out = Tensor::new_zeros(out_shape, self.dtype.clone(), &self.device)?;
        
        // Ensure indices are available as i32
        let indices_f32 = indices.to_numpy_f32_vec();
        let indices_i32: Vec<i32> = indices_f32.into_iter().map(|f| f as i32).collect();

        if self.device.starts_with("vulkan") {
            let indices_u8 = bytemuck::cast_slice::<i32, u8>(&indices_i32);
            let (weight_raw, _) = self.get_slice_raw_bytes();
            let (out_raw, _) = out.get_slice_raw_mut_bytes();
            crate::backend::execute_index_select_into(
                weight_raw,
                indices_u8,
                out_raw,
                num_indices as u32,
                feature_len as u32,
                self.dtype.clone(),
            );
            return Ok(out);
        }

        match self.dtype {
            DataType::F32 => crate::cpu::index_select_f32(self.get_slice_raw_f32().0, &indices_i32, out.get_slice_raw_mut_f32().0, feature_len),
            DataType::F16 => crate::cpu::index_select_f16(self.get_slice_raw_f16().0, &indices_i32, out.get_slice_raw_mut_f16().0, feature_len),
            DataType::BF16 => crate::cpu::index_select_bf16(self.get_slice_raw_bf16().0, &indices_i32, out.get_slice_raw_mut_bf16().0, feature_len),
            DataType::Int8 => crate::cpu::index_select_i8(self.get_slice_raw_i8().0, &indices_i32, out.get_slice_raw_mut_i8().0, feature_len),
            DataType::BitNet2 | DataType::BitNet1_6 => return Err(pyo3::exceptions::PyNotImplementedError::new_err("BitNet index_select not implemented")),
        };
        
        Ok(out)
    }
}
