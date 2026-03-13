use pyo3::prelude::*;
use numpy::{ToPyArray, PyArrayMethods};
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
                }
            }
        }
    }

    pub fn get_slice_raw_f32(&self) -> (&[f32], usize) {
        match &self.storage {
            Storage::F32(v) => (v.as_slice(), v.len()),
            _ => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_f32(&mut self) -> (&mut [f32], usize) {
        match &mut self.storage {
            Storage::F32(v) => { let l = v.len(); (v.as_mut_slice(), l) },
            _ => (&mut [], 0),
        }
    }

    pub fn get_slice_raw_f16(&self) -> (&[half::f16], usize) {
        match &self.storage {
            Storage::F16(v) => (v.as_slice(), v.len()),
            _ => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_f16(&mut self) -> (&mut [half::f16], usize) {
        match &mut self.storage {
            Storage::F16(v) => { let l = v.len(); (v.as_mut_slice(), l) },
            _ => (&mut [], 0),
        }
    }

    pub fn get_slice_raw_bf16(&self) -> (&[half::bf16], usize) {
        match &self.storage {
            Storage::BF16(v) => (v.as_slice(), v.len()),
            _ => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_bf16(&mut self) -> (&mut [half::bf16], usize) {
        match &mut self.storage {
            Storage::BF16(v) => { let l = v.len(); (v.as_mut_slice(), l) },
            _ => (&mut [], 0),
        }
    }

    pub fn get_slice_raw_i8(&self) -> (&[i8], usize) {
        match &self.storage {
            Storage::Int8(v) => (v.as_slice(), v.len()),
            _ => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_i8(&mut self) -> (&mut [i8], usize) {
        match &mut self.storage {
            Storage::Int8(v) => { let l = v.len(); (v.as_mut_slice(), l) },
            _ => (&mut [], 0),
        }
    }

    pub fn get_slice_raw_bytes(&self) -> (&[u8], usize) {
        match &self.storage {
            Storage::F32(v) => (bytemuck::cast_slice(v), v.len() * 4),
            Storage::F16(v) => (bytemuck::cast_slice(v), v.len() * 2),
            Storage::BF16(v) => (bytemuck::cast_slice(v), v.len() * 2),
            Storage::Int8(v) => (bytemuck::cast_slice(v), v.len()),
            Storage::None => (&[], 0),
        }
    }

    pub fn get_slice_raw_mut_bytes(&mut self) -> (&mut [u8], usize) {
        match &mut self.storage {
            Storage::F32(v) => { let l = v.len() * 4; (bytemuck::cast_slice_mut(v), l) },
            Storage::F16(v) => { let l = v.len() * 2; (bytemuck::cast_slice_mut(v), l) },
            Storage::BF16(v) => { let l = v.len() * 2; (bytemuck::cast_slice_mut(v), l) },
            Storage::Int8(v) => { let l = v.len(); (bytemuck::cast_slice_mut(v), l) },
            Storage::None => (&mut [], 0),
        }
    }

    pub fn raw_to_storage(&self, raw: &[u8]) -> Storage {
        match self.dtype {
            DataType::F32 => Storage::F32(bytemuck::cast_slice(raw).to_vec()),
            DataType::F16 => Storage::F16(bytemuck::cast_slice(raw).to_vec()),
            DataType::BF16 => Storage::BF16(bytemuck::cast_slice(raw).to_vec()),
            DataType::Int8 => Storage::Int8(bytemuck::cast_slice(raw).to_vec()),
        }
    }
}
