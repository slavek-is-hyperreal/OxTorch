use pyo3::prelude::*;
use rayon::prelude::*;
use super::{Tensor, DataType, Storage};

impl Tensor {
    pub fn new_from_vec(data: Vec<f32>, shape: Vec<usize>, dtype: DataType, device: &str, name: &str) -> PyResult<Self> {
        let storage = match dtype {
            DataType::F32 => Storage::F32(data),
            _ if device == "cpu" => {
                let size = data.len();
                let n_bytes = if dtype == DataType::BitNet2 { (size + 3) / 4 } 
                              else if dtype == DataType::BitNet1_6 { (size + 4) / 5 } 
                              else if dtype == DataType::I2_S { (size / 64) * 20 }
                              else { size * dtype.size() };
                super::pool::TENSOR_POOL.with(|pool| {
                    let mut p = pool.borrow_mut();
                    let mut buf = p.alloc_raw(n_bytes);
                    let (ptr, _len, cap) = (buf.as_mut_ptr(), buf.len(), buf.capacity());
                    std::mem::forget(buf);
                    
                    match dtype {
                        DataType::F16 => {
                            let mut v = unsafe { Vec::from_raw_parts(ptr as *mut half::f16, size, cap / 2) };
                            for i in 0..size { v[i] = half::f16::from_f32(data[i]); }
                            Storage::F16(v)
                        },
                        DataType::BF16 => {
                            let mut v = unsafe { Vec::from_raw_parts(ptr as *mut half::bf16, size, cap / 2) };
                            for i in 0..size { v[i] = half::bf16::from_f32(data[i]); }
                            Storage::BF16(v)
                        },
                        DataType::Int8 => {
                            let mut v = unsafe { Vec::from_raw_parts(ptr as *mut i8, size, cap) };
                            for i in 0..size { v[i] = data[i] as i8; }
                            Storage::Int8(v)
                        },
                        DataType::BitNet2 | DataType::BitNet1_6 => {
                            let mut v = unsafe { Vec::from_raw_parts(ptr as *mut u8, size, cap) };
                            for i in 0..size { v[i] = data[i] as u8; }
                            Storage::BitNet(v)
                        },
                        DataType::I2_S => {
                            let mut v = unsafe { Vec::from_raw_parts(ptr as *mut u8, size, cap) };
                            for i in 0..size { v[i] = data[i] as u8; }
                            Storage::I2_S(v)
                        },
                        _ => unreachable!(),
                    }
                })
            },
            DataType::F16 => {
                let mut v = vec![half::f16::ZERO; data.len()];
                for i in 0..data.len() { v[i] = half::f16::from_f32(data[i]); }
                Storage::F16(v)
            },
            DataType::BF16 => {
                let mut v = vec![half::bf16::ZERO; data.len()];
                for i in 0..data.len() { v[i] = half::bf16::from_f32(data[i]); }
                Storage::BF16(v)
            },
            DataType::Int8 => {
                let mut v = vec![0i8; data.len()];
                for i in 0..data.len() { v[i] = data[i] as i8; }
                Storage::Int8(v)
            },
            DataType::BitNet2 | DataType::BitNet1_6 | DataType::I2_S => return Err(pyo3::exceptions::PyValueError::new_err("Cannot create quantized raw struct directly from f32. Use from_raw_bytes.")),
        };

        let strides = Self::calculate_default_strides(shape.clone());
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            dtype, 
            device: device.to_owned(), 
            storage, 
            name: name.to_owned(),
            is_transposed: false,
            ssd_engine: None,
        })
    }

    pub fn new(shape: Vec<usize>, dtype: DataType, device: &str, name: &str) -> PyResult<Self> {
        let size: usize = shape.iter().product();
        let n_bytes = size * dtype.size();
        
        let storage = if device == "cpu" {
            super::pool::TENSOR_POOL.with(|pool| {
                let mut p = pool.borrow_mut();
                let mut buf = p.alloc_raw(n_bytes);
                let (ptr, _len, cap) = (buf.as_mut_ptr(), buf.len(), buf.capacity());
                std::mem::forget(buf);
                
                match dtype {
                    DataType::F32 => {
                        let v = unsafe { Vec::from_raw_parts(ptr as *mut f32, size, cap / 4) };
                        Storage::F32(v)
                    },
                    DataType::F16 => {
                        let v = unsafe { Vec::from_raw_parts(ptr as *mut half::f16, size, cap / 2) };
                        Storage::F16(v)
                    },
                    DataType::BF16 => {
                        let v = unsafe { Vec::from_raw_parts(ptr as *mut half::bf16, size, cap / 2) };
                        Storage::BF16(v)
                    },
                    DataType::Int8 => {
                        let v = unsafe { Vec::from_raw_parts(ptr as *mut i8, size, cap) };
                        Storage::Int8(v)
                    },
                    DataType::BitNet2 | DataType::BitNet1_6 | DataType::I2_S => {
                        let n_bytes = if dtype == DataType::BitNet2 { (size + 3) / 4 } 
                                      else if dtype == DataType::I2_S { (size / 64) * 20 }
                                      else { (size + 4) / 5 };
                        let v = unsafe { Vec::from_raw_parts(ptr as *mut u8, n_bytes, cap) };
                        Storage::BitNet(v)
                    },
                }
            })
        } else {
            match dtype {
                DataType::F32 => Storage::F32(vec![0.0; size]),
                DataType::F16 => Storage::F16(vec![half::f16::ZERO; size]),
                DataType::BF16 => Storage::BF16(vec![half::bf16::ZERO; size]),
                DataType::Int8 => Storage::Int8(vec![0; size]),
                DataType::BitNet2 => Storage::BitNet(vec![0; (size + 3) / 4]),
                DataType::BitNet1_6 => Storage::BitNet(vec![0; (size + 4) / 5]),
                DataType::I2_S => Storage::I2_S(vec![0; (size / 64) * 20]),
            }
        };

        let strides = Self::calculate_default_strides(shape.clone());
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            device: device.to_string(), 
            name: name.to_string(),
            is_transposed: false,
            dtype,
            storage, 
            ssd_engine: None 
        })
    }

    pub fn new_zeros(shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        Self::new(shape, dtype, device, "zeros")
    }

    pub fn new_ones(shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        let mut t = Self::new(shape, dtype, device, "ones")?;
        match &mut t.storage {
            Storage::F32(v) => v.fill(1.0),
            Storage::F16(v) => v.fill(half::f16::from_f32(1.0)),
            Storage::BF16(v) => v.fill(half::bf16::from_f32(1.0)),
            Storage::Int8(v) => v.fill(1),
            Storage::BitNet(v) => v.fill(0xFF), // Filled with 'ones' (packed representation depends on bitlayout)
            Storage::I2_S(v) => v.fill(0xFF), // Assuming 0xFF represents 'ones' for I2_S as well
            _ => {},
        }
        Ok(t)
    }

    pub fn new_rand(shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        let size: usize = shape.iter().product();
        let mut data = vec![0.0f32; size];
        let chunk_size = 65536;
        data.par_chunks_mut(chunk_size).enumerate().for_each(|(i, chunk): (usize, &mut [f32])| {
            let seed = 0x1234567890ABCDEF ^ (i as u64);
            let mut rng = crate::prng::Xoshiro256pp::new(seed);
            for val in chunk.iter_mut() { *val = rng.next_f32(); }
        });
        Self::new_from_vec(data, shape, dtype, device, "Rand")
    }

    pub fn new_from_ssd(path: &str, shape: Vec<usize>, dtype: DataType) -> PyResult<Self> {
        let engine = crate::io_uring_engine::DirectIoEngine::new(path, true);
        let strides = Self::calculate_default_strides(shape.clone());
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            device: "ssd".to_string(), 
            name: "SSDMapped".to_string(), 
            is_transposed: false, 
            dtype, 
            storage: Storage::None, 
            ssd_engine: Some(crate::tensor::IoEngineType::ReadOnly(std::sync::Arc::new(engine))) 
        })
    }

    pub fn new_ssd_raw(path: &str, shape: Vec<usize>, dtype: DataType) -> PyResult<Self> {
        let size = shape.iter().product::<usize>();
        let bytes_per_elem = match dtype {
            DataType::F32 => size * 4,
            DataType::F16 | DataType::BF16 => size * 2,
            DataType::Int8 => size,
            DataType::BitNet2 => (size + 3) / 4,
            DataType::BitNet1_6 => (size + 4) / 5,
            DataType::I2_S => (size / 64) * 20, 
        };
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        file.set_len(bytes_per_elem as u64).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let engine = crate::io_uring_engine::DirectIoEngine::new(path, false);
        let strides = Self::calculate_default_strides(shape.clone());
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            device: "ssd".to_string(), 
            name: "SSDResult".to_string(), 
            is_transposed: false, 
            dtype, 
            storage: Storage::None, 
            ssd_engine: Some(crate::tensor::IoEngineType::ReadWrite(std::sync::Arc::new(engine))) 
        })
    }

    pub fn execute_save_ssd(&self, path: &str) -> PyResult<Self> {
        use std::io::Write;
        let (bytes, _) = self.get_slice_raw_bytes();
        let mut file = std::fs::File::create(path).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        file.write_all(bytes).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        file.sync_all().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        
        Self::new_from_ssd(path, self.shape.clone(), self.dtype)
    }

    pub fn new_from_vec_with_shape(data: Vec<f32>, shape: Vec<usize>, dtype: DataType, device: &str, name: &str) -> PyResult<Self> {
        let logical_size: usize = shape.iter().product();
        let expected_bytes = match dtype {
            DataType::BitNet2 => (logical_size + 3) / 4,
            DataType::BitNet1_6 => (logical_size + 4) / 5,
            _ => logical_size * dtype.size(),
        };

        if data.len() < expected_bytes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data size {} is too small for logical shape {:?} with dtype {:?} (expected {} bytes)",
                data.len(), shape, dtype, expected_bytes
            )));
        }

        let storage = match dtype {
            DataType::BitNet2 | DataType::BitNet1_6 => {
                let mut v = vec![0u8; data.len()];
                for i in 0..data.len() { v[i] = data[i] as u8; }
                Storage::BitNet(v)
            },
            _ => return Self::new_from_vec(data, shape, dtype, device, name),
        };

        let strides = Self::calculate_default_strides(shape.clone());
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            dtype, 
            device: device.to_owned(), 
            storage, 
            name: name.to_owned(),
            is_transposed: false,
            ssd_engine: None,
        })
    }

    pub fn from_raw_bytes(data: Vec<u8>, shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        let storage = match dtype {
            DataType::F32 => unsafe {
                let bytes = data;
                let ptr = bytes.as_ptr();
                let len = bytes.len() / 4;
                let cap = bytes.capacity() / 4;
                std::mem::forget(bytes);
                Storage::F32(Vec::from_raw_parts(ptr as *mut f32, len, cap))
            },
            DataType::F16 => unsafe {
                let bytes = data;
                let ptr = bytes.as_ptr();
                let len = bytes.len() / 2;
                let cap = bytes.capacity() / 2;
                std::mem::forget(bytes);
                Storage::F16(Vec::from_raw_parts(ptr as *mut half::f16, len, cap))
            },
            DataType::BF16 => unsafe {
                let bytes = data;
                let ptr = bytes.as_ptr();
                let len = bytes.len() / 2;
                let cap = bytes.capacity() / 2;
                std::mem::forget(bytes);
                Storage::BF16(Vec::from_raw_parts(ptr as *mut half::bf16, len, cap))
            },
            DataType::Int8 => unsafe {
                let bytes = data;
                let ptr = bytes.as_ptr();
                let len = bytes.len();
                let cap = bytes.capacity();
                std::mem::forget(bytes);
                Storage::Int8(Vec::from_raw_parts(ptr as *mut i8, len, cap))
            },
            DataType::BitNet2 | DataType::BitNet1_6 => {
                Storage::BitNet(data)
            },
            DataType::I2_S => {
                Storage::I2_S(data)
            },
        };

        let strides = Self::calculate_default_strides(shape.clone());
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            dtype, 
            device: device.to_owned(), 
            storage, 
            name: "FromRawBytes".to_owned(),
            is_transposed: false,
            ssd_engine: None,
        })
    }
}
