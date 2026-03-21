use pyo3::prelude::*;
use rayon::prelude::*;
use super::{Tensor, DataType, Storage};

impl Tensor {
    pub fn new_from_vec(data: Vec<f32>, shape: Vec<usize>, dtype: DataType, device: &str, name: &str) -> PyResult<Self> {
        let storage = match dtype {
            DataType::F32 => Storage::F32(data),
            DataType::F16 => {
                let mut data_f16 = vec![half::f16::ZERO; data.len()];
                for i in 0..data.len() { data_f16[i] = half::f16::from_f32(data[i]); }
                Storage::F16(data_f16)
            },
            DataType::BF16 => {
                let mut data_bf16 = vec![half::bf16::ZERO; data.len()];
                for i in 0..data.len() { data_bf16[i] = half::bf16::from_f32(data[i]); }
                Storage::BF16(data_bf16)
            },
            DataType::Int8 => {
                let mut data_i8 = vec![0i8; data.len()];
                for i in 0..data.len() { data_i8[i] = data[i] as i8; }
                Storage::Int8(data_i8)
            },
            DataType::Ternary => {
                // BitNet 1.58b Quantization: w = round(clip(w / gamma, -1, 1))
                // gamma = mean(abs(w))
                let gamma = if data.is_empty() { 1.0 } else {
                    data.iter().map(|x| x.abs()).sum::<f32>() / (data.len() as f32)
                }.max(1e-5);
                
                let mut data_t = vec![0i8; data.len()];
                for i in 0..data.len() {
                    let val = (data[i] / gamma).clamp(-1.0, 1.0).round();
                    data_t[i] = val as i8;
                }
                Storage::Ternary(data_t)
            }
        };

        let strides = Self::calculate_default_strides(&shape);
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            dtype, 
            device: device.to_owned(), 
            storage, 
            name: name.to_owned(),
            is_transposed: false,
            mmap_data: None,
        })
    }

    pub fn new(shape: Vec<usize>, dtype: DataType, device: &str, name: &str) -> PyResult<Self> {
        let size: usize = shape.iter().product();
        let storage = match dtype {
            DataType::F32 => Storage::F32(vec![0.0; size]),
            DataType::F16 => Storage::F16(vec![half::f16::ZERO; size]),
            DataType::BF16 => Storage::BF16(vec![half::bf16::ZERO; size]),
            DataType::Int8 => Storage::Int8(vec![0; size]),
            DataType::Ternary => Storage::Ternary(vec![0; size]),
        };
        let strides = Self::calculate_default_strides(&shape);
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            device: device.to_string(), 
            name: name.to_string(),
            is_transposed: false,
            dtype,
            storage, 
            mmap_data: None 
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
            Storage::Ternary(v) => v.fill(1),
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
        let strides = Self::calculate_default_strides(&shape);
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            device: "ssd".to_string(), 
            name: "SSDMapped".to_string(), 
            is_transposed: false, 
            dtype, 
            storage: Storage::None, 
            mmap_data: Some(crate::tensor::IoEngineType::ReadOnly(std::sync::Arc::new(engine))) 
        })
    }

    pub fn new_ssd_raw(path: &str, shape: Vec<usize>, dtype: DataType) -> PyResult<Self> {
        let size = shape.iter().product::<usize>();
        let bytes_per_elem = match dtype {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::Int8 | DataType::Ternary => 1,
        };
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        file.set_len((size * bytes_per_elem) as u64).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let engine = crate::io_uring_engine::DirectIoEngine::new(path, false);
        let strides = Self::calculate_default_strides(&shape);
        Ok(Tensor { 
            shape, 
            strides,
            offset: 0,
            device: "ssd".to_string(), 
            name: "SSDResult".to_string(), 
            is_transposed: false, 
            dtype, 
            storage: Storage::None, 
            mmap_data: Some(crate::tensor::IoEngineType::ReadWrite(std::sync::Arc::new(engine))) 
        })
    }
}
