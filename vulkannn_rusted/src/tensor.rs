use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, ToPyArray};
use numpy::ndarray::Array;
use rayon::prelude::*;
use half::{f16, bf16};
use crate::io_uring_engine::DirectIoEngine;
use crate::buf_pool::BufPool;
use std::sync::Arc;

#[derive(Clone)]
pub enum IoEngineType {
    ReadOnly(Arc<DirectIoEngine>),
    ReadWrite(Arc<DirectIoEngine>),
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[pyclass(eq, eq_int)]
pub enum DataType {
    F32,
    F16,
    BF16,
    Int8,
}

/// Backbone data storage for the Tensor engine.
/// Encapsulates the multi-precision vectors (F32, F16, BF16) or SSD-mapped handles.
#[derive(Clone)]
pub enum Storage {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    BF16(Vec<half::bf16>),
    Int8(Vec<i8>),
    None,
}

/// A multi-precision, multi-device, stateful Tensor object.
/// Bridges Rust's high-speed core with Python's ease of use via PyO3.
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Tensor {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get, set)]
    pub device: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get)]
    pub is_transposed: bool,
    #[pyo3(get)]
    pub dtype: DataType,
    pub storage: Storage,
    pub mmap_data: Option<IoEngineType>,
}


#[pymethods]
impl Tensor {
    #[new]
    #[pyo3(signature = (data=None, shape=None, dtype=DataType::F32, device="auto", name="Tensor"))]
    #[allow(unused_variables)]
    fn new(data: Option<PyReadonlyArrayDyn<'_, f32>>, shape: Option<Vec<usize>>, dtype: DataType, device: &str, name: &str) -> PyResult<Self> {
        let (storage, final_shape) = if let Some(arr) = data {
            let nd_arr = arr.as_array();
            let vec = nd_arr.as_slice().unwrap_or(&[]).to_vec();
            let (storage, final_shape) = match dtype {
                DataType::F32 => (Storage::F32(vec), nd_arr.shape().to_vec()),
                DataType::F16 => {
                    let mut f16_vec = vec![half::f16::ZERO; vec.len()];
                    crate::avx_swar::convert_f32_to_f16(&vec, &mut f16_vec);
                    (Storage::F16(f16_vec), nd_arr.shape().to_vec())
                },
                DataType::BF16 => {
                    let mut bf16_vec = vec![half::bf16::ZERO; vec.len()];
                    crate::avx_swar::convert_f32_to_bf16(&vec, &mut bf16_vec);
                    (Storage::BF16(bf16_vec), nd_arr.shape().to_vec())
                },
                DataType::Int8 => {
                    let mut int8_vec = vec![0i8; vec.len()];
                    vec.par_iter().zip(int8_vec.par_iter_mut()).for_each(|(&s, d)| *d = s as i8);
                    (Storage::Int8(int8_vec), nd_arr.shape().to_vec())
                }
            };
            (storage, final_shape)
        } else if let Some(s) = shape {
            let size = s.iter().product();
            match dtype {
                DataType::F32 => (Storage::F32(vec![0.0; size]), s),
                DataType::F16 => (Storage::F16(vec![half::f16::ZERO; size]), s),
                DataType::BF16 => (Storage::BF16(vec![half::bf16::ZERO; size]), s),
                DataType::Int8 => (Storage::Int8(vec![0i8; size]), s),
            }
        } else {
            return Err(PyValueError::new_err("Must provide either data or shape"));
        };
        Ok(Tensor { shape: final_shape, device: device.to_string(), name: name.to_string(), is_transposed: false, dtype, storage, mmap_data: None })
    }

    #[staticmethod]
    #[pyo3(signature = (path, shape, dtype=DataType::F32))]
    fn from_ssd(path: &str, shape: Vec<usize>, dtype: DataType) -> PyResult<Self> {
        let file = std::fs::File::open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let metadata = file.metadata().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let bytes_per_elem = match dtype {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::Int8 => 1,
        };
        let expected_size = shape.iter().product::<usize>() * bytes_per_elem;
        if metadata.len() < expected_size as u64 {
            return Err(PyValueError::new_err(format!("File size mismatch: expected at least {} bytes, found {}", expected_size, metadata.len())));
        }
        let engine = DirectIoEngine::new(path, true);
        Ok(Tensor { shape, device: "ssd".to_string(), name: "SSDMapped".to_string(), is_transposed: false, dtype, storage: Storage::None, mmap_data: Some(IoEngineType::ReadOnly(Arc::new(engine))) })
    }

    #[staticmethod]
    #[pyo3(signature = (path, shape, dtype=DataType::F32))]
    fn new_ssd(path: &str, shape: Vec<usize>, dtype: DataType) -> PyResult<Self> {
        let size = shape.iter().product::<usize>();
        let bytes_per_elem = match dtype {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::Int8 => 1,
        };
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        file.set_len((size * bytes_per_elem) as u64).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let engine = DirectIoEngine::new(path, false);
        Ok(Tensor { shape, device: "ssd".to_string(), name: "SSDResult".to_string(), is_transposed: false, dtype, storage: Storage::None, mmap_data: Some(IoEngineType::ReadWrite(Arc::new(engine))) })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=DataType::F32, device="cpu"))]
    fn zeros(shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        Self::new(None, Some(shape), dtype, device, "Zeros")
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=DataType::F32, device="cpu"))]
    fn ones(shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        let mut t = Self::new(None, Some(shape.clone()), dtype, device, "Ones")?;
        if device == "cpu" {
            match dtype {
                DataType::F32 => t.get_slice_raw_mut_f32().0.fill(1.0),
                DataType::F16 => t.get_slice_raw_mut_f16().0.fill(half::f16::from_f32(1.0)),
                DataType::BF16 => t.get_slice_raw_mut_bf16().0.fill(half::bf16::from_f32(1.0)),
                DataType::Int8 => t.get_slice_raw_mut_i8().0.fill(1i8),
            }
        } else {
            let mut cpu_t = Self::ones(shape, dtype, "cpu")?;
            t.storage = cpu_t.storage;
        }
        Ok(t)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, fill_value, dtype=DataType::F32, device="cpu"))]
    fn full(shape: Vec<usize>, fill_value: f32, dtype: DataType, device: &str) -> PyResult<Self> {
        let mut t = Self::new(None, Some(shape.clone()), dtype, device, "Full")?;
        if device == "cpu" {
            match dtype {
                DataType::F32 => t.get_slice_raw_mut_f32().0.fill(fill_value),
                DataType::F16 => t.get_slice_raw_mut_f16().0.fill(half::f16::from_f32(fill_value)),
                DataType::BF16 => t.get_slice_raw_mut_bf16().0.fill(half::bf16::from_f32(fill_value)),
                DataType::Int8 => t.get_slice_raw_mut_i8().0.fill(fill_value as i8),
            }
        } else {
            let mut cpu_t = Self::full(shape, fill_value, dtype, "cpu")?;
            t.storage = cpu_t.storage;
        }
        Ok(t)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=DataType::F32, device="cpu"))]
    fn rand<'py>(py: Python<'py>, shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        let np = py.import_bound("numpy.random")?;
        let arr = np.call_method1("rand", pyo3::types::PyTuple::new_bound(py, &shape))?;
        let arr_f32 = arr.call_method1("astype", ("float32",))?;
        let readonly: PyReadonlyArrayDyn<f32> = arr_f32.extract()?;
        Self::new(Some(readonly), None, dtype, device, "Rand")
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=DataType::F32, device="cpu"))]
    fn randn<'py>(py: Python<'py>, shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        let np = py.import_bound("numpy.random")?;
        let arr = np.call_method1("randn", pyo3::types::PyTuple::new_bound(py, &shape))?;
        let arr_f32 = arr.call_method1("astype", ("float32",))?;
        let readonly: PyReadonlyArrayDyn<f32> = arr_f32.extract()?;
        Self::new(Some(readonly), None, dtype, device, "RandN")
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
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
                    crate::avx_swar::convert_f16_to_f32(slice, &mut vec);
                    vec
                }
            },
            DataType::BF16 => {
                if self.is_ssd() { self.load_to_f32_vec_msts() } else {
                    let (slice, _) = self.get_slice_raw_bf16();
                    let mut vec = vec![0.0; slice.len()];
                    crate::avx_swar::convert_bf16_to_f32(slice, &mut vec);
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
        Ok(Array::from_shape_vec(self.shape.clone(), vec).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray_bound(py))
    }

    pub fn to_numpy_f32_vec(&self) -> Vec<f32> {
        if self.is_ssd() { self.load_to_f32_vec_msts() } else {
            match self.dtype {
                DataType::F32 => self.get_slice_raw_f32().0.to_vec(),
                DataType::F16 => {
                    let (s, _) = self.get_slice_raw_f16();
                    let mut v = vec![0.0; s.len()];
                    crate::avx_swar::convert_f16_to_f32(s, &mut v);
                    v
                },
                DataType::BF16 => {
                    let (s, _) = self.get_slice_raw_bf16();
                    let mut v = vec![0.0; s.len()];
                    crate::avx_swar::convert_bf16_to_f32(s, &mut v);
                    v
                },
                DataType::Int8 => {
                    let (s, _) = self.get_slice_raw_i8();
                    s.iter().map(|&x| x as f32).collect()
                }
            }
        }
    }

    fn to_numpy_no_copy<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, numpy::PyArrayDyn<f32>>, bool)> {
        match self.dtype {
            DataType::F32 => {
                if self.is_ssd() {
                    return Ok((self.to_numpy(py)?, false));
                }
                let (slice, is_ssd) = self.get_slice_raw_f32();
                let shape = self.shape.clone();
                let array = unsafe { numpy::ndarray::ArrayViewD::from_shape_ptr(shape, slice.as_ptr()) };
                Ok((array.to_pyarray_bound(py), is_ssd))
            },
            DataType::F16 => {
                if self.is_ssd() { return Ok((self.to_numpy(py)?, false)); }
                let (slice, _) = self.get_slice_raw_f16();
                let mut v = vec![0.0; slice.len()];
                crate::avx_swar::convert_f16_to_f32(slice, &mut v);
                let arr = Array::from_shape_vec(self.shape.clone(), v).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray_bound(py);
                Ok((arr, false))
            },
            DataType::BF16 => {
                if self.is_ssd() { return Ok((self.to_numpy(py)?, false)); }
                let (slice, _) = self.get_slice_raw_bf16();
                let mut v = vec![0.0; slice.len()];
                crate::avx_swar::convert_bf16_to_f32(slice, &mut v);
                let arr = Array::from_shape_vec(self.shape.clone(), v).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray_bound(py);
                Ok((arr, false))
            },
            DataType::Int8 => {
                Ok((self.to_numpy(py)?, false))
            }
        }
    }

    fn __repr__(&self) -> String { format!("Tensor(shape={:?}, dtype={:?}, device='{}', ssd={})", self.shape, self.dtype, self.device, self.mmap_data.is_some()) }

    // --- SHAPE MANIPULATION ---
    fn reshape(&self, new_shape: Vec<i64>) -> PyResult<Tensor> {
        let total_elements = self.shape.iter().product::<usize>();
        let mut target_shape = Vec::new();
        let mut inferred_dim = None;
        let mut cur_elements = 1;
        
        for (i, &dim) in new_shape.iter().enumerate() {
            if dim == -1 {
                if inferred_dim.is_some() {
                    return Err(PyValueError::new_err("Only one dimension can be inferred (-1)"));
                }
                inferred_dim = Some(i);
                target_shape.push(0); // placeholder
            } else if dim <= 0 {
                return Err(PyValueError::new_err(format!("Invalid shape dimension {}", dim)));
            } else {
                target_shape.push(dim as usize);
                cur_elements *= dim as usize;
            }
        }
        
        if let Some(idx) = inferred_dim {
            if total_elements % cur_elements != 0 {
                return Err(PyValueError::new_err("Cannot infer shape securely: elements not divisible"));
            }
            target_shape[idx] = total_elements / cur_elements;
        } else {
            let target_total = target_shape.iter().product::<usize>();
            if target_total != total_elements {
                return Err(PyValueError::new_err(format!("Shape mismatch: tensor has {} elements, target shape requires {}", total_elements, target_total)));
            }
        }

        let mut out = self.clone();
        out.shape = target_shape;
        Ok(out)
    }

    fn view(&self, new_shape: Vec<i64>) -> PyResult<Tensor> {
        self.reshape(new_shape)
    }

    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, dim: Option<i64>) -> PyResult<Tensor> {
        let mut out = self.clone();
        let mut new_shape = Vec::new();
        if let Some(d) = dim {
            let d_idx = if d < 0 { out.shape.len() as i64 + d } else { d };
            if d_idx < 0 || d_idx >= out.shape.len() as i64 {
                return Err(PyValueError::new_err("Dimension out of range"));
            }
            for (i, &s) in out.shape.iter().enumerate() {
                if i == d_idx as usize {
                    if s != 1 {
                        new_shape.push(s); // PyTorch does not squeeze if dim != 1
                    }
                } else {
                    new_shape.push(s);
                }
            }
        } else {
            // Squeeze all size 1 dimensions
            for &s in out.shape.iter() {
                if s != 1 {
                    new_shape.push(s);
                }
            }
        }
        out.shape = new_shape;
        Ok(out)
    }

    fn unsqueeze(&self, dim: i64) -> PyResult<Tensor> {
        let mut out = self.clone();
        let d = if dim < 0 { out.shape.len() as i64 + dim + 1 } else { dim }; // +1 for insertion
        if d < 0 || d > out.shape.len() as i64 {
            return Err(PyValueError::new_err("Dimension out of range"));
        }
        out.shape.insert(d as usize, 1);
        Ok(out)
    }

    #[pyo3(signature = (start_dim=0, end_dim=-1))]
    fn flatten(&self, start_dim: i64, end_dim: i64) -> PyResult<Tensor> {
        let len = self.shape.len() as i64;
        let s = if start_dim < 0 { len + start_dim } else { start_dim };
        let e = if end_dim < 0 { len + end_dim } else { end_dim };
        
        if s < 0 || s >= len || e < 0 || e >= len || s > e {
            return Err(PyValueError::new_err(format!("Invalid flatten dims: start={}, end={}, len={}", start_dim, end_dim, len)));
        }
        
        let mut out = self.clone();
        let mut new_shape = out.shape[0..(s as usize)].to_vec();
        let flat_size = out.shape[(s as usize)..=(e as usize)].iter().product();
        new_shape.push(flat_size);
        new_shape.extend_from_slice(&out.shape[(e as usize + 1)..]);
        out.shape = new_shape;
        Ok(out)
    }

    // --- ELEMENT-WISE ---
    fn __add__(&self, other: &Tensor) -> PyResult<Tensor> {
        self.check_shape(other)?;
        if self.device == "cpu" && other.device == "cpu" {
            if self.dtype == DataType::F32 {
                let (a, _) = self.get_slice_raw_f32();
                let (b, _) = other.get_slice_raw_f32();
                let mut res = vec![0.0; a.len()];
                res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = a + b);
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "AddRes".to_string(), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(res), mmap_data: None })
            } else if self.dtype == DataType::F16 {
                let (a, _) = self.get_slice_raw_f16();
                let (b, _) = other.get_slice_raw_f16();
                let mut res = vec![f16::ZERO; a.len()];
                res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = f16::from_f32(a.to_f32() + b.to_f32()));
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "AddRes".to_string(), is_transposed: false, dtype: DataType::F16, storage: Storage::F16(res), mmap_data: None })
            } else if self.dtype == DataType::BF16 {
                let (a, _) = self.get_slice_raw_bf16();
                let (b, _) = other.get_slice_raw_bf16();
                let mut res = vec![bf16::ZERO; a.len()];
                res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = bf16::from_f32(a.to_f32() + b.to_f32()));
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "AddRes".to_string(), is_transposed: false, dtype: DataType::BF16, storage: Storage::BF16(res), mmap_data: None })
            } else if self.dtype == DataType::Int8 {
                let (a, _) = self.get_slice_raw_i8();
                let (b, _) = other.get_slice_raw_i8();
                let mut res = vec![0i8; a.len()];
                crate::swar_int8::swar_add_int8(a, b, &mut res);
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "AddRes".to_string(), is_transposed: false, dtype: DataType::Int8, storage: Storage::Int8(res), mmap_data: None })
            } else {
                return Err(PyValueError::new_err("Unsupported DataType for CPU Add"));
            }
        } else {
            let (a_raw, _) = self.get_slice_raw_bytes();
            let (b_raw, _) = other.get_slice_raw_bytes();
            let res_raw = crate::backend::execute_add(a_raw, b_raw, self.dtype, self.device == "hybrid");
            let storage = if self.dtype == DataType::F16 {
                Storage::F16(bytemuck::cast_slice(&res_raw).to_vec())
            } else if self.dtype == DataType::BF16 {
                Storage::BF16(bytemuck::cast_slice(&res_raw).to_vec())
            } else if self.dtype == DataType::Int8 {
                Storage::Int8(bytemuck::cast_slice(&res_raw).to_vec())
            } else {
                Storage::F32(bytemuck::cast_slice(&res_raw).to_vec())
            };
            Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "AddRes".to_string(), is_transposed: false, dtype: self.dtype, storage, mmap_data: None })
        }
    }

    fn add_into(&mut self, other: &Tensor, out: &mut Tensor) -> PyResult<()> {
        self.check_shape(other)?;
        if self.device == "cpu" && other.device == "cpu" {
            match self.dtype {
                DataType::F32 => {
                    let (out_slice, _) = out.get_slice_raw_mut_f32();
                    self.add_into_raw_f32(other, out_slice)
                }
                DataType::F16 => {
                    let (out_slice, _) = out.get_slice_raw_mut_f16();
                    let (a, _) = self.get_slice_raw_f16();
                    let (b, _) = other.get_slice_raw_f16();
                    out_slice.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &av), &bv)| *c = half::f16::from_f32(av.to_f32() + bv.to_f32()));
                    Ok(())
                }
                DataType::BF16 => {
                    let (out_slice, _) = out.get_slice_raw_mut_bf16();
                    let (i1, _) = self.get_slice_raw_bf16();
                    let (i2, _) = other.get_slice_raw_bf16();
                    out_slice.par_iter_mut().zip(i1.par_iter()).zip(i2.par_iter()).for_each(|((o, &a), &b)| *o = half::bf16::from_f32(a.to_f32() + b.to_f32()));
                    Ok(())
                }
                DataType::Int8 => {
                    let (out_slice, _) = out.get_slice_raw_mut_i8();
                    let (i1, _) = self.get_slice_raw_i8();
                    let (i2, _) = other.get_slice_raw_i8();
                    crate::swar_int8::swar_add_int8(i1, i2, out_slice);
                    Ok(())
                }
            }
        } else {
            let (a, _) = self.get_slice_raw_bytes();
            let (b, _) = other.get_slice_raw_bytes();
            let (out_raw, _) = out.get_slice_raw_mut_bytes();
            crate::backend::execute_add_into(a, b, out_raw, self.dtype, self.device == "hybrid", false);
            Ok(())
        }
    }

    fn __mul__(&self, other: &Tensor) -> PyResult<Tensor> {
        self.elementwise_op(other, "mul")
    }
    fn __sub__(&self, other: &Tensor) -> PyResult<Tensor> {
        self.elementwise_op(other, "sub")
    }
    fn __truediv__(&self, other: &Tensor) -> PyResult<Tensor> {
        self.elementwise_op(other, "div")
    }

    fn mul_scalar(&self, val: f32) -> PyResult<Tensor> {
        self.scalar_op(val, "mul_scalar")
    }
    fn add_scalar(&self, val: f32) -> PyResult<Tensor> {
        self.scalar_op(val, "add_scalar")
    }
    fn sub_scalar(&self, val: f32) -> PyResult<Tensor> {
        self.scalar_op(val, "sub_scalar")
    }
    fn div_scalar(&self, val: f32) -> PyResult<Tensor> {
        self.scalar_op(val, "div_scalar")
    }

    fn relu(&self) -> PyResult<Tensor> { self.unary_op("relu", 0.0, 0.0) }
    fn relu_into(&mut self, out: &mut Tensor) -> PyResult<()> { self.act_into("relu", 0.0, 0.0, out) }
    fn sigmoid_into(&mut self, out: &mut Tensor) -> PyResult<()> { self.act_into("sigmoid", 0.0, 0.0, out) }
    fn silu_into(&mut self, out: &mut Tensor) -> PyResult<()> { self.act_into("silu", 0.0, 0.0, out) }

    fn gelu(&self) -> PyResult<Tensor> { self.unary_op("gelu", 0.0, 0.0) }
    fn gelu_into(&mut self, out: &mut Tensor) -> PyResult<()> { self.act_into("gelu", 0.0, 0.0, out) }
    fn leaky_relu(&self, negative_slope: f32) -> PyResult<Tensor> { self.unary_op("leaky_relu", negative_slope, 0.0) }
    fn leaky_relu_into(&mut self, negative_slope: f32, out: &mut Tensor) -> PyResult<()> { self.act_into("leaky_relu", negative_slope, 0.0, out) }
    fn elu(&self, alpha: f32) -> PyResult<Tensor> { self.unary_op("elu", alpha, 0.0) }
    fn elu_into(&mut self, alpha: f32, out: &mut Tensor) -> PyResult<()> { self.act_into("elu", alpha, 0.0, out) }
    fn tanh(&self) -> PyResult<Tensor> { self.unary_op("tanh", 0.0, 0.0) }
    fn tanh_into(&mut self, out: &mut Tensor) -> PyResult<()> { self.act_into("tanh", 0.0, 0.0, out) }
    fn clamp(&self, min: f32, max: f32) -> PyResult<Tensor> { self.unary_op("clamp", min, max) }
    fn clamp_into(&mut self, min: f32, max: f32, out: &mut Tensor) -> PyResult<()> { self.act_into("clamp", min, max, out) }

    /// Sum reduction. Returns a scalar-shaped Tensor (1 element).
    #[pyo3(signature = (dim=None))]
    fn sum(&self, dim: Option<i64>) -> PyResult<Tensor> { self.reduce("sum", dim) }
    /// Mean reduction.
    #[pyo3(signature = (dim=None))]
    fn mean(&self, dim: Option<i64>) -> PyResult<Tensor> { self.reduce("mean", dim) }
    /// Max reduction.
    #[pyo3(signature = (dim=None))]
    fn max_val(&self, dim: Option<i64>) -> PyResult<Tensor> { self.reduce("max", dim) }
    /// Min reduction.
    #[pyo3(signature = (dim=None))]
    fn min_val(&self, dim: Option<i64>) -> PyResult<Tensor> { self.reduce("min", dim) }

    /// Softmax activation along specified dimension.
    #[pyo3(signature = (dim))]
    fn softmax(&self, dim: i64) -> PyResult<Tensor> { self.apply_softmax(dim, false) }
    /// Log-Softmax activation. Numerically stable.
    #[pyo3(signature = (dim))]
    fn log_softmax(&self, dim: i64) -> PyResult<Tensor> { self.apply_softmax(dim, true) }

    fn act_into(&mut self, op: &str, param1: f32, param2: f32, out: &mut Tensor) -> PyResult<()> {
        if self.device == "cpu" {
            match self.dtype {
                DataType::F32 => {
                    let (out_slice, _) = out.get_slice_raw_mut_f32();
                    let (i_s, _) = self.get_slice_raw_f32();
                    Self::act_into_raw_f32(i_s, op, param1, param2, out_slice)
                }
                DataType::F16 => {
                    let (out_slice, _) = out.get_slice_raw_mut_f16();
                    let (i_s, _) = self.get_slice_raw_f16();
                    match op {
                        "relu" => out_slice.par_iter_mut().zip(i_s.par_iter()).for_each(|(o, &i)| *o = if i > half::f16::ZERO { i } else { half::f16::ZERO }),
                        _ => panic!("Unsupported F16 CPU op into: {}", op),
                    }
                    Ok(())
                }
                DataType::BF16 => {
                    let (out_slice, _) = out.get_slice_raw_mut_bf16();
                    let (i_s, _) = self.get_slice_raw_bf16();
                    Self::act_into_raw_bf16(i_s, op, param1, param2, out_slice);
                    Ok(())
                },
                DataType::Int8 => {
                    let (out_slice, _) = out.get_slice_raw_mut_i8();
                    let (i_s, _) = self.get_slice_raw_i8();
                    if op == "relu" {
                        out_slice.par_iter_mut().zip(i_s.par_iter()).for_each(|(o, &i)| *o = if i > 0 { i } else { 0 });
                    }
                    Ok(())
                }
            }
        } else {
            let (input_raw, _) = self.get_slice_raw_bytes();
            let (res_raw, _) = out.get_slice_raw_mut_bytes();
            crate::backend::execute_activation_into(input_raw, op, param1, param2, res_raw, self.dtype, self.device == "hybrid", false);
            Ok(())
        }
    }

    fn sigmoid(&self) -> PyResult<Tensor> { self.unary_op("sigmoid", 0.0, 0.0) }
    fn silu(&self) -> PyResult<Tensor> { self.unary_op("silu", 0.0, 0.0) }

    /// Matrix Multiplication (dot product).
    /// Dispatches to `matrixmultiply::sgemm` on CPU or specialized SPIR-V kernel on GPU.
    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 { return Err(PyValueError::new_err("2D required")); }
        let m = self.shape[0]; let k = self.shape[1]; let k2 = other.shape[0]; let n = other.shape[1];
        if k != k2 { return Err(PyValueError::new_err("K mismatch")); }
        if self.dtype != other.dtype { return Err(PyValueError::new_err("DType mismatch")); }

        let rsa = if self.is_transposed { 1 } else { k as isize };
        let csa = if self.is_transposed { self.shape[0] as isize } else { 1 };
        let rsb = if other.is_transposed { 1 } else { n as isize };
        let csb = if other.is_transposed { other.shape[0] as isize } else { 1 };

        if self.device == "cpu" {
            let dtype = self.dtype;
            crate::streaming::init_budgets();
            let safe_ram = crate::streaming::BUDGETS.get().unwrap().lock().unwrap().l2_ram_max_bytes;
            let bytes_per_elem = if dtype == DataType::F32 { 4 } else if dtype == DataType::Int8 { 1 } else { 2 };
            let sz = (m*k + k*n + m*n) * bytes_per_elem;

            let storage = if sz < safe_ram {
                match dtype {
                    DataType::F32 => {
                        let mut out = vec![0.0; m * n];
                        let a_ssd = self.is_ssd();
                        let b_ssd = other.is_ssd();
                        if a_ssd || b_ssd {
                            let ar = if a_ssd { self.load_to_f32_vec_msts() } else { self.get_slice_raw_f32().0.to_vec() };
                            let br = if b_ssd { other.load_to_f32_vec_msts() } else { other.get_slice_raw_f32().0.to_vec() };
                            self.cpu_sgemm_f32(&ar, &br, &mut out, m, k, n, rsa, csa, rsb, csb);
                        } else {
                            let (a, _) = self.get_slice_raw_f32();
                            let (b, _) = other.get_slice_raw_f32();
                            self.cpu_sgemm_f32(a, b, &mut out, m, k, n, rsa, csa, rsb, csb);
                        }
                        Storage::F32(out)
                    },
                    DataType::F16 => {
                        let mut out = vec![half::f16::ZERO; m * n];
                        let a_f32 = if self.is_ssd() { self.load_to_f32_vec_msts() } else {
                            let mut v = vec![0.0; self.shape.iter().product()];
                            crate::avx_swar::convert_f16_to_f32(self.get_slice_raw_f16().0, &mut v);
                            v
                        };
                        let b_f32 = if other.is_ssd() { other.load_to_f32_vec_msts() } else {
                            let mut v = vec![0.0; other.shape.iter().product()];
                            crate::avx_swar::convert_f16_to_f32(other.get_slice_raw_f16().0, &mut v);
                            v
                        };
                        let mut res_f32 = BufPool::get(m * n);
                        res_f32.resize(m * n, 0.0f32);
                        self.cpu_sgemm_f32(&a_f32, &b_f32, &mut res_f32, m, k, n, rsa, csa, rsb, csb);
                        out.par_iter_mut().zip(res_f32.par_iter()).for_each(|(o, &x)| *o = half::f16::from_f32(x));
                        BufPool::put(res_f32); // Return intermediate buffer to pool
                        Storage::F16(out)
                    },
                    DataType::BF16 => {
                        let mut out = vec![half::bf16::ZERO; m * n];
                        let a_f32 = if self.is_ssd() { self.load_to_f32_vec_msts() } else {
                            let mut v = vec![0.0; self.shape.iter().product()];
                            crate::avx_swar::convert_bf16_to_f32(self.get_slice_raw_bf16().0, &mut v);
                            v
                        };
                        let b_f32 = if other.is_ssd() { other.load_to_f32_vec_msts() } else {
                            let mut v = vec![0.0; other.shape.iter().product()];
                            crate::avx_swar::convert_bf16_to_f32(other.get_slice_raw_bf16().0, &mut v);
                            v
                        };
                        let mut res_f32 = BufPool::get(m * n);
                        res_f32.resize(m * n, 0.0f32);
                        self.cpu_sgemm_f32(&a_f32, &b_f32, &mut res_f32, m, k, n, rsa, csa, rsb, csb);
                        out.par_iter_mut().zip(res_f32.par_iter()).for_each(|(o, &x)| *o = half::bf16::from_f32(x));
                        BufPool::put(res_f32); // Return intermediate buffer to pool
                        Storage::BF16(out)
                    },
                    DataType::Int8 => {
                        // For now fallback Int8 matmul to F32 then cast back
                        let mut out = vec![0i8; m * n];
                        let a_f32 = self.to_numpy_f32_vec();
                        let b_f32 = other.to_numpy_f32_vec();
                        let mut res_f32 = BufPool::get(m * n);
                        res_f32.resize(m * n, 0.0f32);
                        self.cpu_sgemm_f32(&a_f32, &b_f32, &mut res_f32, m, k, n, rsa, csa, rsb, csb);
                        out.par_iter_mut().zip(res_f32.par_iter()).for_each(|(o, &x)| *o = x as i8);
                        BufPool::put(res_f32);
                        Storage::Int8(out)
                    }
                }
            } else {
                match dtype {
                    DataType::F32 => {
                        let mut out = vec![0.0; m * n];
                        let (a, _) = self.get_slice_raw_f32();
                        let (b, _) = other.get_slice_raw_f32();
                        self.cpu_sgemm_streamed_f32(a, b, &mut out, m, k, n, rsa, csa, rsb, csb);
                        Storage::F32(out)
                    },
                    DataType::F16 | DataType::BF16 | DataType::Int8 => {
                        let is_f16 = dtype == DataType::F16;
                        let is_int8 = dtype == DataType::Int8;
                        let bytes_per_elem = if is_int8 { 1 } else { 2 };
                        let mut out_bytes = vec![0u8; m * n * bytes_per_elem];
                        let (a_bytes, _) = self.get_slice_raw_bytes();
                        let (b_bytes, _) = other.get_slice_raw_bytes();
                        
                        // Streamed F16/BF16/Int8: For now fallback to F32 via streaming then convert back
                        // Optimizing this for streaming BF16 next.
                        let mut res_f32 = BufPool::get(m * n);
                        res_f32.resize(m * n, 0.0f32);
                        let (a_f32, b_f32) = if is_f16 {
                            let a_f16: &[half::f16] = bytemuck::cast_slice(a_bytes);
                            let b_f16: &[half::f16] = bytemuck::cast_slice(b_bytes);
                            (a_f16.par_iter().map(|&x| x.to_f32()).collect::<Vec<_>>(),
                             b_f16.par_iter().map(|&x| x.to_f32()).collect::<Vec<_>>())
                        } else if is_int8 {
                            let a_int8: &[i8] = bytemuck::cast_slice(a_bytes);
                            let b_int8: &[i8] = bytemuck::cast_slice(b_bytes);
                            (a_int8.par_iter().map(|&x| x as f32).collect::<Vec<_>>(),
                             b_int8.par_iter().map(|&x| x as f32).collect::<Vec<_>>())
                        } else { // BF16
                            let a_bf16: &[half::bf16] = bytemuck::cast_slice(a_bytes);
                            let b_bf16: &[half::bf16] = bytemuck::cast_slice(b_bytes);
                            (a_bf16.par_iter().map(|&x| x.to_f32()).collect::<Vec<_>>(),
                             b_bf16.par_iter().map(|&x| x.to_f32()).collect::<Vec<_>>())
                        };
                        
                        self.cpu_sgemm_streamed_f32(&a_f32, &b_f32, &mut res_f32, m, k, n, rsa, csa, rsb, csb);
                        
                        if is_f16 {
                            let out_f16: &mut [half::f16] = bytemuck::cast_slice_mut(&mut out_bytes);
                            out_f16.par_iter_mut().zip(res_f32.par_iter()).for_each(|(o, &x)| *o = half::f16::from_f32(x));
                            Storage::F16(out_f16.to_vec())
                        } else if is_int8 {
                            let out_int8: &mut [i8] = bytemuck::cast_slice_mut(&mut out_bytes);
                            out_int8.par_iter_mut().zip(res_f32.par_iter()).for_each(|(o, &x)| *o = x as i8);
                            Storage::Int8(out_int8.to_vec())
                        } else { // BF16
                            let out_bf16: &mut [half::bf16] = bytemuck::cast_slice_mut(&mut out_bytes);
                            out_bf16.par_iter_mut().zip(res_f32.par_iter()).for_each(|(o, &x)| *o = half::bf16::from_f32(x));
                            Storage::BF16(out_bf16.to_vec())
                        }
                    }
                }
            };
            return Ok(Tensor { shape: vec![m, n], device: self.device.clone(), name: "MatMulRes".to_string(), is_transposed: false, dtype, storage, mmap_data: None });
        } else {
            let (a_raw, _) = self.get_slice_raw_bytes();
            let (b_raw, _) = other.get_slice_raw_bytes();
            let res_raw = crate::backend::execute_matmul(a_raw, b_raw, m as u32, k as u32, n as u32, self.dtype, self.device == "hybrid");
            let storage = if self.dtype == DataType::F16 {
                Storage::F16(bytemuck::cast_slice(&res_raw).to_vec())
            } else if self.dtype == DataType::BF16 {
                Storage::BF16(bytemuck::cast_slice(&res_raw).to_vec())
            } else if self.dtype == DataType::Int8 {
                Storage::Int8(bytemuck::cast_slice(&res_raw).to_vec())
            } else {
                Storage::F32(bytemuck::cast_slice(&res_raw).to_vec())
            };
            Ok(Tensor {
                shape: vec![m, n],
                device: self.device.clone(),
                name: "MatMulRes".to_string(),
                is_transposed: false,
                dtype: self.dtype,
                storage,
                mmap_data: None
            })
        }
    }
    fn transpose(&self) -> PyResult<Tensor> {
        if self.shape.len() != 2 { return Err(PyValueError::new_err("2D required for transpose")); }
        Ok(Tensor { 
            shape: vec![self.shape[1], self.shape[0]], 
            device: self.device.clone(), 
            name: format!("{}.T", self.name),
            is_transposed: !self.is_transposed,
            dtype: self.dtype,
            storage: self.storage.clone(), 
            mmap_data: self.mmap_data.clone(), 
        })
    }
}

// Internal Logic
impl Tensor {
    pub fn is_ssd(&self) -> bool {
        self.mmap_data.is_some()
    }

    fn check_shape(&self, other: &Tensor) -> PyResult<()> {
        if self.shape != other.shape { return Err(PyValueError::new_err("Shape mismatch")); }
        if self.dtype != other.dtype { return Err(PyValueError::new_err("DType mismatch")); }
        Ok(())
    }

    /// Generic elementwise binary op: mul / sub / div (CPU+Vulkan+Hybrid, all dtypes)
    fn elementwise_op(&self, other: &Tensor, op: &str) -> PyResult<Tensor> {
        self.check_shape(other)?;
        let op_name = match op { "mul" => "MulRes", "sub" => "SubRes", "div" => "DivRes", _ => "ElemRes" };
        if self.device == "cpu" && other.device == "cpu" {
            if self.dtype == DataType::F32 {
                let (a, _) = self.get_slice_raw_f32();
                let (b, _) = other.get_slice_raw_f32();
                let mut res = BufPool::get(a.len());
                match op {
                    "mul" => res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = a * b),
                    "sub" => res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = a - b),
                    "div" => res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = a / b),
                    _ => {}
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: op_name.to_string(), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(res), mmap_data: None })
            } else if self.dtype == DataType::F16 {
                let (a, _) = self.get_slice_raw_f16();
                let (b, _) = other.get_slice_raw_f16();
                let mut res = vec![f16::ZERO; a.len()];
                match op {
                    "mul" => res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = f16::from_f32(a.to_f32() * b.to_f32())),
                    "sub" => res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = f16::from_f32(a.to_f32() - b.to_f32())),
                    "div" => res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = f16::from_f32(a.to_f32() / b.to_f32())),
                    _ => {}
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: op_name.to_string(), is_transposed: false, dtype: DataType::F16, storage: Storage::F16(res), mmap_data: None })
            } else if self.dtype == DataType::BF16 {
                let (a, _) = self.get_slice_raw_bf16();
                let (b, _) = other.get_slice_raw_bf16();
                let mut res = vec![bf16::ZERO; a.len()];
                match op {
                    "mul" => res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = bf16::from_f32(a.to_f32() * b.to_f32())),
                    "sub" => res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = bf16::from_f32(a.to_f32() - b.to_f32())),
                    "div" => res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = bf16::from_f32(a.to_f32() / b.to_f32())),
                    _ => {}
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: op_name.to_string(), is_transposed: false, dtype: self.dtype, storage: Storage::BF16(res), mmap_data: None })
            } else if self.dtype == DataType::Int8 {
                let (a, _) = self.get_slice_raw_i8();
                let (b, _) = other.get_slice_raw_i8();
                let mut res = vec![0i8; a.len()];
                match op {
                    "mul" => { for i in 0..a.len() { res[i] = a[i].wrapping_mul(b[i]); } },
                    "sub" => { for i in 0..a.len() { res[i] = a[i].wrapping_sub(b[i]); } },
                    "div" => { for i in 0..a.len() { res[i] = if b[i] != 0 { a[i] / b[i] } else { 0 }; } },
                    _ => {}
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: op_name.to_string(), is_transposed: false, dtype: DataType::Int8, storage: Storage::Int8(res), mmap_data: None })
            } else {
                return Err(PyValueError::new_err("Unsupported DataType for CPU Add"));
            }
        } else {
            let (a_raw, _) = self.get_slice_raw_bytes();
            let (b_raw, _) = other.get_slice_raw_bytes();
            let res_raw = crate::backend::execute_elementwise(a_raw, b_raw, op, self.dtype, self.device == "hybrid");
            let storage = if self.dtype == DataType::F16 {
                Storage::F16(bytemuck::cast_slice(&res_raw).to_vec())
            } else if self.dtype == DataType::BF16 {
                Storage::BF16(bytemuck::cast_slice(&res_raw).to_vec())
            } else if self.dtype == DataType::Int8 {
                Storage::Int8(bytemuck::cast_slice(&res_raw).to_vec())
            } else {
                Storage::F32(bytemuck::cast_slice(&res_raw).to_vec())
            };
            Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: op_name.to_string(), is_transposed: false, dtype: self.dtype, storage, mmap_data: None })
        }
    }

    /// Broadcast scalar op: mul_scalar / add_scalar / sub_scalar / div_scalar (CPU only for now)
    fn scalar_op(&self, val: f32, op: &str) -> PyResult<Tensor> {
        if self.device == "cpu" {
            if self.dtype == DataType::F32 {
                let (input, _) = self.get_slice_raw_f32();
                let mut res = BufPool::get(input.len());
                match op {
                    "mul_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = i * val),
                    "add_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = i + val),
                    "sub_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = i - val),
                    "div_scalar" => { let inv = 1.0 / val; res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = i * inv); },
                    _ => {}
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "ScalarRes".to_string(), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(res), mmap_data: None })
            } else if self.dtype == DataType::F16 {
                let (input, _) = self.get_slice_raw_f16();
                let mut res = vec![f16::ZERO; input.len()];
                match op {
                    "mul_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = f16::from_f32(i.to_f32() * val)),
                    "add_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = f16::from_f32(i.to_f32() + val)),
                    "sub_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = f16::from_f32(i.to_f32() - val)),
                    "div_scalar" => { let inv = 1.0 / val; res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = f16::from_f32(i.to_f32() * inv)); },
                    _ => {}
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "ScalarRes".to_string(), is_transposed: false, dtype: DataType::F16, storage: Storage::F16(res), mmap_data: None })
            } else if self.dtype == DataType::BF16 {
                let (input, _) = self.get_slice_raw_bf16();
                let mut res = vec![bf16::ZERO; input.len()];
                match op {
                    "mul_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() * val)),
                    "add_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() + val)),
                    "sub_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() - val)),
                    "div_scalar" => { let inv = 1.0 / val; res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() * inv)); },
                    _ => {}
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "ScalarRes".to_string(), is_transposed: false, dtype: DataType::BF16, storage: Storage::BF16(res), mmap_data: None })
            } else if self.dtype == DataType::Int8 {
                let (input, _) = self.get_slice_raw_i8();
                let mut res = vec![0i8; input.len()];
                match op {
                    "mul_scalar" => crate::swar_int8::swar_scale_int8(input, val as i8, &mut res),
                    "add_scalar" => { for i in 0..input.len() { res[i] = input[i].wrapping_add(val as i8); } },
                    "sub_scalar" => { for i in 0..input.len() { res[i] = input[i].wrapping_sub(val as i8); } },
                    "div_scalar" => { for i in 0..input.len() { res[i] = if val as i8 != 0 { input[i] / (val as i8) } else { 0 }; } },
                    _ => {}
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "ScalarRes".to_string(), is_transposed: false, dtype: DataType::Int8, storage: Storage::Int8(res), mmap_data: None })
            } else {
                return Err(PyValueError::new_err("Unsupported DataType for CPU Add"));
            }
        } else {
            // Vulkan path: treat scalar as a second buffer of same size for now (fallback to CPU)
            // Full Vulkan scalar push-constant path is Sprint 4
            self.scalar_op_cpu_fallback(val, op)
        }
    }

    fn scalar_op_cpu_fallback(&self, val: f32, op: &str) -> PyResult<Tensor> {
        // For vulkan/hybrid devices, fall back to CPU for scalar ops until Sprint 4 Vulkan shader
        if self.dtype == DataType::F32 {
            let (input, _) = self.get_slice_raw_f32();
            let mut res = BufPool::get(input.len());
            match op {
                "mul_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = i * val),
                "add_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = i + val),
                "sub_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = i - val),
                "div_scalar" => { let inv = 1.0 / val; res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = i * inv); },
                _ => {}
            }
            Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "ScalarRes".to_string(), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(res), mmap_data: None })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Scalar ops on Vulkan non-F32 not yet implemented"))
        }
    }

    fn apply_softmax(&self, dim: i64, is_log: bool) -> PyResult<Tensor> {
        let d_usize = if dim < 0 { (self.shape.len() as i64 + dim) as usize } else { dim as usize };
        if d_usize >= self.shape.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid dimension"));
        }

        if self.device == "cpu" {
            let stride_d: usize = self.shape[d_usize + 1..].iter().product();
            let size_d = self.shape[d_usize];
            let out_len: usize = self.shape.iter().product();
            let outer_stride = stride_d * size_d;
            let num_outer = if outer_stride == 0 { 0 } else { out_len / outer_stride };

            match self.dtype {
                DataType::F32 => {
                    let (input, _) = self.get_slice_raw_f32();
                    let mut out = vec![0.0; out_len];
                    let out_ptr = out.as_mut_ptr() as usize;
                    (0..num_outer).into_par_iter().for_each(|n| {
                        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
                        for k in 0..stride_d {
                            let mut max_val = f32::NEG_INFINITY;
                            for m in 0..size_d {
                                let idx = n * outer_stride + m * stride_d + k;
                                max_val = f32::max(max_val, input[idx]);
                            }
                            let mut sum_exp = 0.0;
                            for m in 0..size_d {
                                let idx = n * outer_stride + m * stride_d + k;
                                sum_exp += f32::exp(input[idx] - max_val);
                            }
                            if is_log {
                                let log_sum = f32::ln(sum_exp);
                                for m in 0..size_d {
                                    let idx = n * outer_stride + m * stride_d + k;
                                    out_slice[idx] = input[idx] - max_val - log_sum;
                                }
                            } else {
                                let inv_sum = 1.0 / sum_exp;
                                for m in 0..size_d {
                                    let idx = n * outer_stride + m * stride_d + k;
                                    out_slice[idx] = f32::exp(input[idx] - max_val) * inv_sum;
                                }
                            }
                        }
                    });
                    Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: if is_log { "LogSoftmax".to_string() } else { "Softmax".to_string() }, is_transposed: false, dtype: DataType::F32, storage: Storage::F32(out), mmap_data: None })
                }
                DataType::F16 => {
                    let (input, _) = self.get_slice_raw_f16();
                    let mut out = vec![f16::ZERO; out_len];
                    let out_ptr = out.as_mut_ptr() as usize;
                    (0..num_outer).into_par_iter().for_each(|n| {
                        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f16, out_len) };
                        for k in 0..stride_d {
                            let mut max_val = f32::NEG_INFINITY;
                            for m in 0..size_d {
                                max_val = f32::max(max_val, input[n * outer_stride + m * stride_d + k].to_f32());
                            }
                            let mut sum_exp = 0.0;
                            for m in 0..size_d {
                                sum_exp += f32::exp(input[n * outer_stride + m * stride_d + k].to_f32() - max_val);
                            }
                            if is_log {
                                let log_sum = f32::ln(sum_exp);
                                for m in 0..size_d {
                                    let idx = n * outer_stride + m * stride_d + k;
                                    out_slice[idx] = f16::from_f32(input[idx].to_f32() - max_val - log_sum);
                                }
                            } else {
                                let inv_sum = 1.0 / sum_exp;
                                for m in 0..size_d {
                                    let idx = n * outer_stride + m * stride_d + k;
                                    out_slice[idx] = f16::from_f32(f32::exp(input[idx].to_f32() - max_val) * inv_sum);
                                }
                            }
                        }
                    });
                    Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: if is_log { "LogSoftmax".to_string() } else { "Softmax".to_string() }, is_transposed: false, dtype: DataType::F16, storage: Storage::F16(out), mmap_data: None })
                }
                DataType::BF16 => {
                    let (input, _) = self.get_slice_raw_bf16();
                    let mut out = vec![bf16::ZERO; out_len];
                    let out_ptr = out.as_mut_ptr() as usize;
                    (0..num_outer).into_par_iter().for_each(|n| {
                        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut bf16, out_len) };
                        for k in 0..stride_d {
                            let mut max_val = f32::NEG_INFINITY;
                            for m in 0..size_d {
                                max_val = f32::max(max_val, input[n * outer_stride + m * stride_d + k].to_f32());
                            }
                            let mut sum_exp = 0.0;
                            for m in 0..size_d {
                                sum_exp += f32::exp(input[n * outer_stride + m * stride_d + k].to_f32() - max_val);
                            }
                            if is_log {
                                let log_sum = f32::ln(sum_exp);
                                for m in 0..size_d {
                                    let idx = n * outer_stride + m * stride_d + k;
                                    out_slice[idx] = bf16::from_f32(input[idx].to_f32() - max_val - log_sum);
                                }
                            } else {
                                let inv_sum = 1.0 / sum_exp;
                                for m in 0..size_d {
                                    let idx = n * outer_stride + m * stride_d + k;
                                    out_slice[idx] = bf16::from_f32(f32::exp(input[idx].to_f32() - max_val) * inv_sum);
                                }
                            }
                        }
                    });
                    Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: if is_log { "LogSoftmax".to_string() } else { "Softmax".to_string() }, is_transposed: false, dtype: DataType::BF16, storage: Storage::BF16(out), mmap_data: None })
                },
                DataType::Int8 => {
                    // Fallback for Int8 softmax to F32 then cast back
                    let (input, _) = self.get_slice_raw_i8();
                    let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
                    let mut out_f32 = vec![0.0; out_len];
                    let out_ptr_f32 = out_f32.as_mut_ptr() as usize;
                    (0..num_outer).into_par_iter().for_each(|n| {
                        let out_slice_f32 = unsafe { std::slice::from_raw_parts_mut(out_ptr_f32 as *mut f32, out_len) };
                        for k in 0..stride_d {
                            let mut max_val = f32::NEG_INFINITY;
                            for m in 0..size_d {
                                let idx = n * outer_stride + m * stride_d + k;
                                max_val = f32::max(max_val, input_f32[idx]);
                            }
                            let mut sum_exp = 0.0;
                            for m in 0..size_d {
                                let idx = n * outer_stride + m * stride_d + k;
                                sum_exp += f32::exp(input_f32[idx] - max_val);
                            }
                            if is_log {
                                let log_sum = f32::ln(sum_exp);
                                for m in 0..size_d {
                                    let idx = n * outer_stride + m * stride_d + k;
                                    out_slice_f32[idx] = input_f32[idx] - max_val - log_sum;
                                }
                            } else {
                                let inv_sum = 1.0 / sum_exp;
                                for m in 0..size_d {
                                    let idx = n * outer_stride + m * stride_d + k;
                                    out_slice_f32[idx] = f32::exp(input_f32[idx] - max_val) * inv_sum;
                                }
                            }
                        }
                    });
                    let out_int8: Vec<i8> = out_f32.par_iter().map(|&x| x as i8).collect();
                    Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: if is_log { "LogSoftmax".to_string() } else { "Softmax".to_string() }, is_transposed: false, dtype: DataType::Int8, storage: Storage::Int8(out_int8), mmap_data: None })
                }
            }
        } else {
            // Vulkan / Hybrid fallback to CPU initially
            let mut cpu_tensor = self.clone();
            cpu_tensor.device = "cpu".to_string();
            let mut res = cpu_tensor.apply_softmax(dim, is_log)?;
            res.device = self.device.clone();
            Ok(res)
        }
    }

    fn reduce(&self, op: &str, dim: Option<i64>) -> PyResult<Tensor> {
        // --- 1. Fast Path: GPU-accelerated Full Reduction ---
        if self.device != "cpu" && dim.is_none() {
            let (a_raw, _) = self.get_slice_raw_bytes();
            let blocks = crate::backend::execute_reduce(a_raw, op, self.dtype);
            let sum: f32 = blocks.iter().sum();
            let num_total = self.shape.iter().product::<usize>();
            let val = match op {
                "mean" => sum / (num_total as f32).max(1.0),
                "max" => blocks.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                "min" => blocks.iter().cloned().fold(f32::INFINITY, f32::min),
                _ => sum,
            };
            
            // Result metadata (Int8 sum/mean upcasts to F32)
            let is_int8_sum_mean = self.dtype == DataType::Int8 && (op == "sum" || op == "mean");
            let res_dtype = if is_int8_sum_mean { DataType::F32 } else { self.dtype };

            let storage = match res_dtype {
                DataType::F32 => Storage::F32(vec![val]),
                DataType::F16 => Storage::F16(vec![half::f16::from_f32(val)]),
                DataType::BF16 => Storage::BF16(vec![half::bf16::from_f32(val)]),
                DataType::Int8 => Storage::Int8(vec![val as i8]),
                _ => Storage::None,
            };
            return Ok(Tensor { shape: vec![], device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: res_dtype, storage, mmap_data: None });
        }

        // --- 2. CPU / Fallback Path (Supports Axis Reduction) ---
        if self.dtype == DataType::F32 {
            let (input, _) = self.get_slice_raw_f32();
            if let Some(d) = dim {
                let d_usize = if d < 0 { (self.shape.len() as i64 + d) as usize } else { d as usize };
                if d_usize >= self.shape.len() { return Err(PyValueError::new_err("Invalid dimension")); }
                
                let mut out_shape = self.shape.clone();
                out_shape.remove(d_usize);
                if out_shape.is_empty() { out_shape = vec![1]; }

                let stride_d: usize = self.shape[d_usize + 1..].iter().product();
                let size_d = self.shape[d_usize];
                let out_len: usize = out_shape.iter().product();
                let mut out = vec![0.0; out_len];

                out.par_iter_mut().enumerate().for_each(|(i, o)| {
                    let n = i / stride_d;
                    let k = i % stride_d;
                    let mut acc = match op {
                        "max" => f32::NEG_INFINITY,
                        "min" => f32::INFINITY,
                        _ => 0.0,
                    };
                    for m in 0..size_d {
                        let in_idx = n * (stride_d * size_d) + m * stride_d + k;
                        let val = input[in_idx];
                        acc = match op {
                            "max" => f32::max(acc, val),
                            "min" => f32::min(acc, val),
                            _ => acc + val,
                        };
                    }
                    *o = if op == "mean" { acc / (size_d as f32) } else { acc };
                });

                Ok(Tensor { shape: out_shape, device: self.device.clone(), name: format!("{}_dim_res", op), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(out), mmap_data: None })
            } else {
                let val = match op {
                    "sum" => input.par_iter().sum::<f32>(),
                    "mean" => input.par_iter().sum::<f32>() / input.len() as f32,
                    "max" => input.par_iter().cloned().reduce(|| f32::NEG_INFINITY, f32::max),
                    "min" => input.par_iter().cloned().reduce(|| f32::INFINITY, f32::min),
                    _ => panic!("Unsupported reduction op: {}", op),
                };
                Ok(Tensor { shape: vec![], device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(vec![val]), mmap_data: None })
            }
        } else if self.dtype == DataType::F16 {
            let (input, _) = self.get_slice_raw_f16();
            let val = match op {
                "sum" => input.par_iter().map(|&x| x.to_f32()).sum::<f32>(),
                "mean" => input.par_iter().map(|&x| x.to_f32()).sum::<f32>() / input.len() as f32,
                "max" => input.par_iter().map(|&x| x.to_f32()).reduce(|| f32::NEG_INFINITY, f32::max),
                "min" => input.par_iter().map(|&x| x.to_f32()).reduce(|| f32::INFINITY, f32::min),
                _ => panic!("Unsupported reduction op: {}", op),
            };
            Ok(Tensor { shape: vec![], device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::F16, storage: Storage::F16(vec![half::f16::from_f32(val)]), mmap_data: None })
        } else if self.dtype == DataType::BF16 {
            let (input, _) = self.get_slice_raw_bf16();
            let val = match op {
                "sum" => input.par_iter().map(|&x| x.to_f32()).sum::<f32>(),
                "mean" => input.par_iter().map(|&x| x.to_f32()).sum::<f32>() / input.len() as f32,
                "max" => input.par_iter().map(|&x| x.to_f32()).reduce(|| f32::NEG_INFINITY, f32::max),
                "min" => input.par_iter().map(|&x| x.to_f32()).reduce(|| f32::INFINITY, f32::min),
                _ => panic!("Unsupported reduction op: {}", op),
            };
            Ok(Tensor { shape: vec![], device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::BF16, storage: Storage::BF16(vec![half::bf16::from_f32(val)]), mmap_data: None })
        } else if self.dtype == DataType::Int8 {
            let (input, _) = self.get_slice_raw_i8();
            let val = match op {
                "sum" => input.par_iter().map(|&x| x as i32).sum::<i32>() as f32,
                "mean" => (input.par_iter().map(|&x| x as i32).sum::<i32>() as f32) / input.len() as f32,
                "max" => input.par_iter().map(|&x| x as i32).reduce(|| i8::MIN as i32, |a, b| a.max(b)) as f32,
                "min" => input.par_iter().map(|&x| x as i32).reduce(|| i8::MAX as i32, |a, b| a.min(b)) as f32,
                _ => panic!("Unsupported reduction op: {}", op),
            };
            if op == "sum" || op == "mean" {
                Ok(Tensor { shape: vec![], device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(vec![val]), mmap_data: None })
            } else {
                Ok(Tensor { shape: vec![], device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::Int8, storage: Storage::Int8(vec![val as i8]), mmap_data: None })
            }
        } else {
            Err(pyo3::exceptions::PyNotImplementedError::new_err("Reduce not supported for this dtype"))
        }
    }

    fn unary_op(&self, op: &str, param1: f32, param2: f32) -> PyResult<Tensor> {
        if self.is_ssd() {
            return self.unary_op_ssd(op, param1, param2);
        }
        if self.device == "cpu" {
            if self.dtype == DataType::F32 {
                let (input, _) = self.get_slice_raw_f32();
                // BufPool::get reuses warm memory from a previous call — avoids cold OS page fault.
                // This is the primary fix for relu() being 10x slower than PyTorch on small tensors.
                let mut res = BufPool::get(input.len());
                Self::act_into_raw_f32(input, op, param1, param2, &mut res)?;
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(res), mmap_data: None })
            } else if self.dtype == DataType::F16 {
                let (input, _) = self.get_slice_raw_f16();
                let mut res = vec![f16::ZERO; input.len()];
                match op {
                    "relu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = if i.to_f32() > 0.0 { i } else { f16::ZERO }),
                    "sigmoid" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = f16::from_f32(1.0 / (1.0 + (-i.to_f32()).exp()))),
                    "silu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = f16::from_f32(f / (1.0 + (-f).exp())) }),
                    "gelu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = f16::from_f32(0.5 * f * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (f + 0.044715 * f.powi(3))).tanh())) }),
                    "leaky_relu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = f16::from_f32(if f > 0.0 { f } else { f * param1 }) }),
                    "elu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = f16::from_f32(if f > 0.0 { f } else { param1 * (f.exp() - 1.0) }) }),
                    "tanh" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = f16::from_f32(i.to_f32().tanh())),
                    "clamp" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = f16::from_f32(i.to_f32().clamp(param1, param2))),
                    _ => panic!("Unsupported F16 CPU op: {}", op),
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::F16, storage: Storage::F16(res), mmap_data: None })
            } else if self.dtype == DataType::BF16 {
                let (input, _) = self.get_slice_raw_bf16();
                let mut res = vec![bf16::ZERO; input.len()];
                match op {
                    "relu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = if i.to_f32() > 0.0 { i } else { bf16::ZERO }),
                    "sigmoid" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(1.0 / (1.0 + (-i.to_f32()).exp()))),
                    "silu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = bf16::from_f32(f / (1.0 + (-f).exp())) }),
                    "gelu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = bf16::from_f32(0.5 * f * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (f + 0.044715 * f.powi(3))).tanh())) }),
                    "leaky_relu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = bf16::from_f32(if f > 0.0 { f } else { f * param1 }) }),
                    "elu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = bf16::from_f32(if f > 0.0 { f } else { param1 * (f.exp() - 1.0) }) }),
                    "tanh" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32().tanh())),
                    "clamp" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32().clamp(param1, param2))),
                    _ => panic!("Unsupported BF16 CPU op: {}", op),
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::BF16, storage: Storage::BF16(res), mmap_data: None })
            } else if self.dtype == DataType::Int8 {
                let (input, _) = self.get_slice_raw_i8();
                let mut res = vec![0i8; input.len()];
                match op {
                    "relu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = if i > 0 { i } else { 0 }),
                    "clamp" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = i.clamp(param1 as i8, param2 as i8)),
                    "sigmoid" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { *o = (1.0 / (1.0 + (-(i as f32)).exp()) * 127.0) as i8 }),
                    "gelu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| { 
                        let f = i as f32;
                        *o = (0.5 * f * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (f + 0.044715 * f.powi(3))).tanh())) as i8;
                    }),
                    _ => return Err(PyValueError::new_err(format!("Unsupported Int8 CPU op: {}", op))),
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::Int8, storage: Storage::Int8(res), mmap_data: None })
            } else {
                return Err(PyValueError::new_err("Unsupported DataType for CPU Unary Op"));
            }
        } else if self.device == "hybrid" {
            // --- MSTS Tile-Pulling Hybrid Dispatch (Phase 4) ---
            // Divide the tensor into N tiles of ~TILE_ELEMS elements each.
            // An AtomicUsize acts as a shared tile counter (Tagged-Token Dataflow).
            // One GPU dispatcher thread and multiple CPU worker threads race to claim
            // tiles via fetch_add. The fastest resource "eats" the most work.
            let (input_raw, _) = self.get_slice_raw_bytes();
            let bpe_fetch = match self.dtype {
                DataType::F32  => 4,
                DataType::Int8 => 1,
                _ => 2,
            };
            let num_total = input_raw.len() / bpe_fetch;

            // Tile size: ~256K elements (~1MB for F32) aligns with our ZFS block target
            const TILE_ELEMS: usize = 256 * 1024;
            // Vulkan dispatch has a fixed overhead of ~80ms on Bonaire (PCIe staging roundtrip).
            // Only use the GPU dispatcher when the total data is large enough to amortize that cost.
            // Below this threshold, pure CPU SWAR is always faster on this hardware.
            const VULKAN_MIN_ELEMS: usize = 4 * 1024 * 1024; // ~16MB F32 / ~8MB F16

            let num_tiles = num_total.div_ceil(TILE_ELEMS);

            let out_bytes_len = input_raw.len();
            let mut out_raw = vec![0u8; out_bytes_len];

            // We need raw ptrs for cross-thread writes into disjoint non-overlapping slices.
            // Safety: tiles are non-overlapping; the AtomicUsize guarantees each tile is
            // claimed at most once. We never write the same byte from two threads.
            let input_ptr = input_raw.as_ptr() as usize; // Send as usize (no &ref lifetime issue)
            let output_ptr = out_raw.as_mut_ptr() as usize;
            let total_len  = out_bytes_len;
            let dtype      = self.dtype;
            let op_str     = op.to_string();

            let tile_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

            rayon::scope(|s| {
                // GPU dispatcher thread: only spawn if tensor is above the Vulkan break-even point.
                // Below VULKAN_MIN_ELEMS the PCIe staging overhead dominates; pure CPU is faster.
                if num_total >= VULKAN_MIN_ELEMS {
                let counter_gpu = tile_counter.clone();
                let op_gpu = op_str.clone();
                s.spawn(move |_| {

                    loop {
                        let tile_id = counter_gpu.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if tile_id >= num_tiles { break; }
                        let elem_start = tile_id * TILE_ELEMS;
                        let elem_count = (TILE_ELEMS).min(num_total - elem_start);
                        let bpe = match dtype {
                            DataType::F32 => 4usize,
                            DataType::Int8 => 1usize,
                            _ => 2usize,
                        };
                        // Safety: raw ptrs are valid for `total_len` bytes
                        let in_sl  = unsafe { std::slice::from_raw_parts    (input_ptr  as *const u8, total_len) };
                        let out_sl = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut   u8, total_len) };
                        let _ = bpe; // suppress lint
                        crate::backend::execute_activation_chunked(in_sl, out_sl, elem_start, elem_count, &op_gpu, param1, param2, dtype);
                    }
                });
                } // end if num_total >= VULKAN_MIN_ELEMS

                // Bug #5 fix: was a single CPU worker — only 1 core used!
                // Now spawn N CPU workers (one per Rayon thread available), racing on the
                // tile_counter. This matches the work-stealing intent of the MSTS architecture.
                let num_cpu_workers = rayon::current_num_threads();
                for _ in 0..num_cpu_workers {
                let cpu_tiles = num_tiles; // upper bound visible to Rayon workers below
                let counter_cpu = tile_counter.clone();
                let op_cpu = op_str.clone();
                s.spawn(move |_| {
                    loop {
                        let tile_id = counter_cpu.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if tile_id >= cpu_tiles { break; }
                        let elem_start = tile_id * TILE_ELEMS;
                        let elem_count = (TILE_ELEMS).min(num_total - elem_start);
                        let bpe = match dtype {
                            DataType::F32 => 4usize,
                            DataType::Int8 => 1usize,
                            _ => 2usize,
                        };
                        let in_sl  = unsafe { std::slice::from_raw_parts    (input_ptr  as *const u8, total_len) };
                        let out_sl = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut   u8, total_len) };
                        match dtype {
                            DataType::F32 => {
                                let in_f  = unsafe { std::slice::from_raw_parts    (in_sl.as_ptr()  .add(elem_start * bpe) as *const f32, elem_count) };
                                let out_f = unsafe { std::slice::from_raw_parts_mut(out_sl.as_mut_ptr().add(elem_start * bpe) as *mut f32, elem_count) };
                                match op_cpu.as_str() {
                                    "relu"    => out_f.iter_mut().zip(in_f.iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 }),
                                    "sigmoid" => out_f.iter_mut().zip(in_f.iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp())),
                                    "silu"    => out_f.iter_mut().zip(in_f.iter()).for_each(|(o, &i)| *o = i / (1.0 + (-i).exp())),
                                    "gelu"    => out_f.iter_mut().zip(in_f.iter()).for_each(|(o, &i)| *o = 0.5 * i * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (i + 0.044715 * i.powi(3))).tanh())),
                                    "leaky_relu" => out_f.iter_mut().zip(in_f.iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { i * param1 }),
                                    "elu"     => out_f.iter_mut().zip(in_f.iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { param1 * (i.exp() - 1.0) }),
                                    "tanh"    => out_f.iter_mut().zip(in_f.iter()).for_each(|(o, &i)| *o = i.tanh()),
                                    "clamp"   => out_f.iter_mut().zip(in_f.iter()).for_each(|(o, &i)| *o = i.clamp(param1, param2)),
                                    _ => panic!("Unsupported CPU op in hybrid tile worker"),
                                }
                            },
                            DataType::F16 => {
                                let in_h  = unsafe { std::slice::from_raw_parts    (in_sl.as_ptr()  .add(elem_start * bpe) as *const half::f16, elem_count) };
                                let out_h = unsafe { std::slice::from_raw_parts_mut(out_sl.as_mut_ptr().add(elem_start * bpe) as *mut half::f16, elem_count) };
                                match op_cpu.as_str() {
                                    "relu" => out_h.iter_mut().zip(in_h.iter()).for_each(|(o, &i)| *o = if i > half::f16::ZERO { i } else { half::f16::ZERO }),
                                    "sigmoid" => out_h.iter_mut().zip(in_h.iter()).for_each(|(o, &i)| *o = half::f16::from_f32(1.0 / (1.0 + (-i.to_f32()).exp()))),
                                    "silu" => out_h.iter_mut().zip(in_h.iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = half::f16::from_f32(f / (1.0 + (-f).exp())) }),
                                    "gelu" => out_h.iter_mut().zip(in_h.iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = half::f16::from_f32(0.5 * f * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (f + 0.044715 * f.powi(3))).tanh())) }),
                                    "leaky_relu" => out_h.iter_mut().zip(in_h.iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = half::f16::from_f32(if f > 0.0 { f } else { f * param1 }) }),
                                    "elu" => out_h.iter_mut().zip(in_h.iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = half::f16::from_f32(if f > 0.0 { f } else { param1 * (f.exp() - 1.0) }) }),
                                    "tanh" => out_h.iter_mut().zip(in_h.iter()).for_each(|(o, &i)| *o = half::f16::from_f32(i.to_f32().tanh())),
                                    "clamp" => out_h.iter_mut().zip(in_h.iter()).for_each(|(o, &i)| *o = half::f16::from_f32(i.to_f32().clamp(param1, param2))),
                                    _ => panic!("Unsupported F16 hybrid tile op"),
                                }
                            },
                            DataType::BF16 => {
                                let in_b  = unsafe { std::slice::from_raw_parts    (in_sl.as_ptr()  .add(elem_start * bpe) as *const half::bf16, elem_count) };
                                let out_b = unsafe { std::slice::from_raw_parts_mut(out_sl.as_mut_ptr().add(elem_start * bpe) as *mut half::bf16, elem_count) };
                                match op_cpu.as_str() {
                                    "relu" => out_b.iter_mut().zip(in_b.iter()).for_each(|(o, &i)| *o = if i > half::bf16::ZERO { i } else { half::bf16::ZERO }),
                                    "sigmoid" => out_b.iter_mut().zip(in_b.iter()).for_each(|(o, &i)| *o = half::bf16::from_f32(1.0 / (1.0 + (-i.to_f32()).exp()))),
                                    "silu" => out_b.iter_mut().zip(in_b.iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = half::bf16::from_f32(f / (1.0 + (-f).exp())) }),
                                    "gelu" => out_b.iter_mut().zip(in_b.iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = half::bf16::from_f32(0.5 * f * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (f + 0.044715 * f.powi(3))).tanh())) }),
                                    "leaky_relu" => out_b.iter_mut().zip(in_b.iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = half::bf16::from_f32(if f > 0.0 { f } else { f * param1 }) }),
                                    "elu" => out_b.iter_mut().zip(in_b.iter()).for_each(|(o, &i)| { let f = i.to_f32(); *o = half::bf16::from_f32(if f > 0.0 { f } else { param1 * (f.exp() - 1.0) }) }),
                                    "tanh" => out_b.iter_mut().zip(in_b.iter()).for_each(|(o, &i)| *o = half::bf16::from_f32(i.to_f32().tanh())),
                                    "clamp" => out_b.iter_mut().zip(in_b.iter()).for_each(|(o, &i)| *o = half::bf16::from_f32(i.to_f32().clamp(param1, param2))),
                                    _ => panic!("Unsupported BF16 hybrid tile op"),
                                }
                            },
                            DataType::Int8 => {
                                let in_i  = unsafe { std::slice::from_raw_parts    (in_sl.as_ptr()  .add(elem_start * bpe) as *const i8, elem_count) };
                                let out_i = unsafe { std::slice::from_raw_parts_mut(out_sl.as_mut_ptr().add(elem_start * bpe) as *mut i8, elem_count) };
                                    match op_cpu.as_str() {
                                        "relu" => out_i.iter_mut().zip(in_i.iter()).for_each(|(o, &i)| *o = if i > 0 { i } else { 0 }),
                                        "clamp" => out_i.iter_mut().zip(in_i.iter()).for_each(|(o, &i)| *o = i.clamp(param1 as i8, param2 as i8)),
                                        "mul_scalar" => out_i.iter_mut().zip(in_i.iter()).for_each(|(o, &i)| *o = i.wrapping_mul(param1 as i8)),
                                        "add_scalar" => out_i.iter_mut().zip(in_i.iter()).for_each(|(o, &i)| *o = i.wrapping_add(param1 as i8)),
                                        "sigmoid" => out_i.iter_mut().zip(in_i.iter()).for_each(|(o, &i)| { *o = (1.0 / (1.0 + (-(i as f32)).exp()) * 127.0) as i8 }),
                                        "gelu" => out_i.iter_mut().zip(in_i.iter()).for_each(|(o, &i)| { 
                                            let f = i as f32;
                                            *o = (0.5 * f * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (f + 0.044715 * f.powi(3))).tanh())) as i8;
                                        }),
                                        _ => {
                                            for k in 0..elem_count {
                                                out_i[k] = in_i[k];
                                            }
                                        }
                                    }
                            }
                        }
                    }
                });
                } // end for cpu workers
            });

            let storage = match dtype {
                DataType::F16 => Storage::F16(bytemuck::cast_slice(&out_raw).to_vec()),
                DataType::BF16 => Storage::BF16(bytemuck::cast_slice(&out_raw).to_vec()),
                DataType::Int8 => Storage::Int8(bytemuck::cast_slice(&out_raw).to_vec()),
                _ => Storage::F32(bytemuck::cast_slice(&out_raw).to_vec()),
            };
            Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: self.dtype, storage, mmap_data: None })

        } else {
            // Pure Vulkan path (device == "vulkan")
            let (input_raw, _) = self.get_slice_raw_bytes();
            let res_raw = crate::backend::execute_activation(input_raw, op, param1, param2, self.dtype, false);
            let storage = match self.dtype {
                DataType::F16 => Storage::F16(bytemuck::cast_slice(&res_raw).to_vec()),
                DataType::BF16 => Storage::BF16(bytemuck::cast_slice(&res_raw).to_vec()),
                DataType::Int8 => Storage::Int8(bytemuck::cast_slice(&res_raw).to_vec()),
                _ => Storage::F32(bytemuck::cast_slice(&res_raw).to_vec()),
            };
            Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: self.dtype, storage, mmap_data: None })
        }
    }

    fn unary_op_ssd(&self, op: &str, param1: f32, param2: f32) -> PyResult<Tensor> {
        let res_path = format!("{}_{}.ssd", self.name, op);
        let res_tensor = Self::new_ssd(&res_path, self.shape.clone(), self.dtype)?;
        
        let engine_in = match self.mmap_data.as_ref().unwrap() {
            IoEngineType::ReadOnly(e) => e.clone(),
            IoEngineType::ReadWrite(e) => e.clone(),
        };
        let engine_out = match res_tensor.mmap_data.as_ref().unwrap() {
            IoEngineType::ReadWrite(e) => e.clone(),
            _ => unreachable!(),
        };
        
        let bytes_per_elem = match self.dtype {
            DataType::F32  => 4,
            DataType::Int8 => 1,
            _ => 2,
        };
        let total_elements = self.shape.iter().product::<usize>();
        let total_bytes = (total_elements * bytes_per_elem) as u64;
        
        // MSTS Ring Buffer (8 tiles of 1MB each)
        let ring_size = 8;
        let scheduler = crate::crook_scheduler::CrookScheduler::new(ring_size);
        
        // Start Workers
        let r_sched = scheduler.clone();
        let w_sched = scheduler.clone();
        let r_handle = crate::crook_scheduler::CrookScheduler::start_read_worker(r_sched, engine_in, total_bytes);
        let w_handle = crate::crook_scheduler::CrookScheduler::start_write_worker(w_sched, engine_out, total_bytes);
        
        // Compute Worker (Main Thread)
        let mut offset = 0;
        let mut tile_idx = 0;
        while offset < total_bytes {
            let tile = &scheduler.ring[tile_idx];
            
            // Spin until tile is READY_FOR_COMPUTE
            while tile.state.compare_exchange(
                crate::crook_scheduler::TILE_READY_FOR_COMPUTE,
                crate::crook_scheduler::TILE_COMPUTING,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed
            ).is_err() {
                std::hint::spin_loop();
            }
            
            let bytes_in_tile = std::cmp::min(1048576, (total_bytes - offset) as usize);
            let payload = unsafe { &mut *tile.payload.get() };
            
            // Apply operation based on dtype
            match self.dtype {
                DataType::F32 => {
                    let slice = bytemuck::cast_slice_mut::<u8, f32>(&mut payload[..bytes_in_tile]);
                    Self::act_into_raw_parallel_f32(slice, op, param1, param2);
                },
                DataType::F16 => {
                    let slice = bytemuck::cast_slice_mut::<u8, half::f16>(&mut payload[..bytes_in_tile]);
                    slice.par_iter_mut().for_each(|x| {
                        if op == "relu" && x.to_f32() < 0.0 { *x = half::f16::ZERO; }
                    });
                },
                DataType::BF16 => {
                    let slice = bytemuck::cast_slice_mut::<u8, half::bf16>(&mut payload[..bytes_in_tile]);
                    slice.par_iter_mut().for_each(|x| {
                        if op == "relu" && x.to_f32() < 0.0 { *x = half::bf16::ZERO; }
                    });
                },
                DataType::Int8 => {
                    let slice = bytemuck::cast_slice_mut::<u8, i8>(&mut payload[..bytes_in_tile]);
                    slice.par_iter_mut().for_each(|x| {
                        if op == "relu" && *x < 0 { *x = 0; }
                    });
                }
            }
            
            tile.state.store(crate::crook_scheduler::TILE_READY_FOR_WRITE, std::sync::atomic::Ordering::Release);
            
            offset += bytes_in_tile as u64;
            tile_idx = (tile_idx + 1) % ring_size;
        }
        
        r_handle.join().unwrap();
        w_handle.join().unwrap();
        
        Ok(res_tensor)
    }

    fn act_into_raw_bf16(slice: &[half::bf16], op: &str, param1: f32, param2: f32, out_slice: &mut [half::bf16]) {
        out_slice.par_iter_mut().zip(slice.par_iter()).for_each(|(o, &i)| {
            let f = i.to_f32();
            *o = half::bf16::from_f32(match op {
                "relu" => if f > 0.0 { f } else { 0.0 },
                "sigmoid" => 1.0 / (1.0 + (-f).exp()),
                "silu" => f / (1.0 + (-f).exp()),
                "tanh" => f.tanh(),
                "clamp" => f.clamp(param1, param2),
                _ => f,
            });
        });
    }

    fn act_into_raw_parallel_f32(slice: &mut [f32], op: &str, param1: f32, param2: f32) {
        match op {
            "relu" => slice.par_iter_mut().for_each(|x| if *x < 0.0 { *x = 0.0; }),
            "sigmoid" => slice.par_iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp())),
            "silu" => slice.par_iter_mut().for_each(|x| *x = *x / (1.0 + (-*x).exp())),
            "gelu" => slice.par_iter_mut().for_each(|x| *x = 0.5 * *x * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (*x + 0.044715 * x.powi(3))).tanh())),
            "leaky_relu" => slice.par_iter_mut().for_each(|x| if *x < 0.0 { *x *= param1; }),
            "elu" => slice.par_iter_mut().for_each(|x| if *x < 0.0 { *x = param1 * (x.exp() - 1.0); }),
            "tanh" => slice.par_iter_mut().for_each(|x| *x = x.tanh()),
            "clamp" => slice.par_iter_mut().for_each(|x| *x = x.clamp(param1, param2)),
            _ => panic!("Unsupported Op: {}", op),
        }
    }

    pub fn get_slice_raw_bytes(&self) -> (&[u8], bool) {
        match &self.storage {
            Storage::F32(v) => (bytemuck::cast_slice(v), false),
            Storage::F16(v) => (bytemuck::cast_slice(v), false),
            Storage::BF16(v) => (bytemuck::cast_slice(v), false),
            Storage::Int8(v) => (bytemuck::cast_slice(v), false),
            Storage::None => {
                if let Some(_) = &self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else {
                    (&[], false)
                }
            }
        }
    }

    pub fn get_slice_raw_f32(&self) -> (&[f32], bool) {
        match &self.storage {
            Storage::F32(v) => (v, false),
            Storage::None => {
                if let Some(_) = &self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else { (&[], false) }
            },
            _ => (&[], false),
        }
    }

    fn get_slice_raw_f16(&self) -> (&[half::f16], bool) {
        match &self.storage {
            Storage::F16(v) => (v, false),
            Storage::None => {
                if let Some(_) = &self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else { (&[], false) }
            },
            _ => (&[], false),
        }
    }

    fn get_slice_raw_bf16(&self) -> (&[half::bf16], bool) {
        match &self.storage {
            Storage::BF16(v) => (v, false),
            Storage::None => {
                if let Some(_) = &self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else { (&[], false) }
            },
            _ => (&[], false),
        }
    }

    pub fn get_slice_raw_i8(&self) -> (&[i8], bool) {
        match &self.storage {
            Storage::Int8(v) => (v.as_slice(), false),
            Storage::None => {
                if let Some(_) = &self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else { (&[], false) }
            },
            _ => (&[], false),
        }
    }

    pub fn get_slice_raw_mut_bytes(&mut self) -> (&mut [u8], bool) {
        match &mut self.storage {
            Storage::F32(v) => (bytemuck::cast_slice_mut(v), false),
            Storage::F16(v) => (bytemuck::cast_slice_mut(v), false),
            Storage::BF16(v) => (bytemuck::cast_slice_mut(v), false),
            Storage::Int8(v) => (bytemuck::cast_slice_mut(v), false),
            Storage::None => {
                if let Some(_) = &mut self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else { panic!("No data") }
            },
        }
    }

    fn get_slice_raw_mut_f32(&mut self) -> (&mut [f32], bool) {
        match &mut self.storage {
            Storage::F32(v) => (v, false),
            Storage::None => {
                if let Some(_) = &self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else { panic!("No data") }
            },
            _ => panic!("Wrong DType"),
        }
    }

    fn get_slice_raw_mut_f16(&mut self) -> (&mut [half::f16], bool) {
        match &mut self.storage {
            Storage::F16(v) => (v, false),
            Storage::None => {
                if let Some(_) = &mut self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else { panic!("No data") }
            },
            _ => panic!("Wrong DType"),
        }
    }

    fn get_slice_raw_mut_bf16(&mut self) -> (&mut [half::bf16], bool) {
        match &mut self.storage {
            Storage::BF16(v) => (v, false),
            Storage::None => {
                if let Some(_) = &mut self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else { panic!("No data") }
            },
            _ => panic!("Wrong DType"),
        }
    }

    pub fn get_slice_raw_mut_i8(&mut self) -> (&mut [i8], bool) {
        match &mut self.storage {
            Storage::Int8(v) => (v.as_mut_slice(), false),
            Storage::None => {
                if let Some(_) = &mut self.mmap_data {
                    panic!("MSTS Architecture: Cannot get flat slice of O_DIRECT SSD tensors. Use chunk streaming.");
                } else { panic!("No data") }
            },
            _ => panic!("Wrong DType"),
        }
    }

    fn add_into_raw_f32(&self, other: &Tensor, out: &mut [f32]) -> PyResult<()> {
        let (a, a_ssd) = self.get_slice_raw_f32();
        let (b, b_ssd) = other.get_slice_raw_f32();
        if !a_ssd && !b_ssd {
            out.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &av), &bv)| *c = av + bv);
        } else {
            let tile = 1024 * 1024 / 4; 
            out.par_chunks_mut(tile).enumerate().for_each(|(i, chunk)| {
                let start = i * tile;
                let asub = &a[start..start + chunk.len()];
                let bsub = &b[start..start + chunk.len()];
                for k in 0..chunk.len() { chunk[k] = asub[k] + bsub[k]; }
            });
        }
        Ok(())
    }

    fn act_into_raw_f32(i_s: &[f32], op: &str, param1: f32, param2: f32, out: &mut [f32]) -> PyResult<()> {
        let total_elements = i_s.len();
        // Use serial execution for very small arrays (L1 cache size, < 32k items)
        // For larger arrays, Rayon overhead is well compensated by utilizing all cores.
        if total_elements < 32_000 {
            // Serial path — AVX1 vmaxps (8 floats/cycle) for relu, scalar fallback for rest
            match op {
                "relu"    => crate::avx_swar::relu_f32(i_s, out),
                "sigmoid" => { for (o, &i) in out.iter_mut().zip(i_s.iter()) { *o = 1.0 / (1.0 + (-i).exp()); } },
                "silu"    => { for (o, &i) in out.iter_mut().zip(i_s.iter()) { *o = i * (1.0 / (1.0 + (-i).exp())); } },
                "gelu"    => { for (o, &i) in out.iter_mut().zip(i_s.iter()) { *o = 0.5 * i * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (i + 0.044715 * i.powi(3))).tanh()); } },
                "leaky_relu" => { for (o, &i) in out.iter_mut().zip(i_s.iter()) { *o = if i > 0.0 { i } else { i * param1 }; } },
                "elu"     => { for (o, &i) in out.iter_mut().zip(i_s.iter()) { *o = if i > 0.0 { i } else { param1 * (i.exp() - 1.0) }; } },
                "tanh"    => { for (o, &i) in out.iter_mut().zip(i_s.iter()) { *o = i.tanh(); } },
                "clamp"   => { for (o, &i) in out.iter_mut().zip(i_s.iter()) { *o = i.clamp(param1, param2); } },
                _ => {}
            }
            } else {
                let tile = 1024 * 1024 / 4;
                out.par_chunks_mut(tile).enumerate().for_each(|(idx, chunk)| {
                    let s = idx * tile;
                    let end = std::cmp::min(s + chunk.len(), i_s.len());
                    if s < end {
                        let isub = &i_s[s..end];
                        let process_len = std::cmp::min(chunk.len(), isub.len());
                        match op {
                            "relu" => crate::avx_swar::relu_f32(isub, &mut chunk[..process_len]),
                            "sigmoid" => for k in 0..process_len { chunk[k] = 1.0 / (1.0 + (-isub[k]).exp()); },
                            "silu" => for k in 0..process_len { chunk[k] = isub[k] * (1.0 / (1.0 + (-isub[k]).exp())); },
                            "gelu" => for k in 0..process_len { chunk[k] = 0.5 * isub[k] * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2 * (isub[k] + 0.044715 * isub[k].powi(3))).tanh()); },
                            "leaky_relu" => for k in 0..process_len { chunk[k] = if isub[k] > 0.0 { isub[k] } else { isub[k] * param1 }; },
                            "elu" => for k in 0..process_len { chunk[k] = if isub[k] > 0.0 { isub[k] } else { param1 * (isub[k].exp() - 1.0) }; },
                            "tanh" => for k in 0..process_len { chunk[k] = isub[k].tanh(); },
                            "clamp" => for k in 0..process_len { chunk[k] = isub[k].clamp(param1, param2); },
                            _ => {}
                        }
                    }
                });
            }
        Ok(())
    }

    /// Extreme I/O MERA-400 architecture for SSD tensors
    pub fn load_to_f32_vec_msts(&self) -> Vec<f32> {
        let engine = match &self.mmap_data {
            Some(crate::tensor::IoEngineType::ReadOnly(e)) => e.clone(),
            Some(crate::tensor::IoEngineType::ReadWrite(e)) => e.clone(),
            None => panic!("Not an SSD tensor"),
        };
        
        // 1MB = 262144 f32s OR 524288 f16s
        let total_elems = self.shape.iter().product::<usize>();
        let mut out = vec![0.0; total_elems];
        let bytes_per_elem = match self.dtype {
            DataType::F32 => 4,
            DataType::Int8 => 1,
            _ => 2,
        };
        let total_bytes = (total_elems * bytes_per_elem) as u64;
        
        let scheduler = crate::crook_scheduler::CrookScheduler::new(8); // 8MB ring
        let io_handle = crate::crook_scheduler::CrookScheduler::start_read_worker(scheduler.clone(), engine, total_bytes);
        
        let mut offset = 0;
        let ring_size = scheduler.ring.len();
        let mut tile_idx = 0;
        
        while offset < total_bytes {
            let tile = &scheduler.ring[tile_idx];
            
            // Spin wait for TILE_READY_FOR_COMPUTE
            while tile.state.compare_exchange(
                crate::crook_scheduler::TILE_READY_FOR_COMPUTE,
                crate::crook_scheduler::TILE_COMPUTING,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed
            ).is_err() {
                std::hint::spin_loop();
            }
            
            let bytes_in_tile = std::cmp::min(1048576, (total_bytes - offset) as usize);
            let payload = unsafe { &*tile.payload.get() };
            
            if self.dtype == DataType::F32 {
                let slice = bytemuck::cast_slice::<u8, f32>(&payload[..bytes_in_tile]);
                let start_idx = (offset / 4) as usize;
                out[start_idx..start_idx + slice.len()].copy_from_slice(slice);
            } else if self.dtype == DataType::F16 {
                let slice = bytemuck::cast_slice::<u8, half::f16>(&payload[..bytes_in_tile]);
                let start_idx = (offset / 2) as usize;
                for (i, val) in slice.iter().enumerate() {
                    out[start_idx + i] = val.to_f32();
                }
            } else if self.dtype == DataType::BF16 {
                let slice = bytemuck::cast_slice::<u8, half::bf16>(&payload[..bytes_in_tile]);
                let start_idx = (offset / 2) as usize;
                for (i, val) in slice.iter().enumerate() {
                    out[start_idx + i] = val.to_f32();
                }
            } else if self.dtype == DataType::Int8 {
                let slice = bytemuck::cast_slice::<u8, i8>(&payload[..bytes_in_tile]);
                let start_idx = offset as usize;
                for (i, val) in slice.iter().enumerate() {
                    out[start_idx + i] = *val as f32;
                }
            }
            
            // Mark as empty to allow PPU to reuse
            tile.state.store(crate::crook_scheduler::TILE_EMPTY, std::sync::atomic::Ordering::Release);
            
            offset += bytes_in_tile as u64;
            tile_idx = (tile_idx + 1) % ring_size;
        }
        
        io_handle.join().unwrap();
        out
    }

    fn cpu_sgemm_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize, rsa: isize, csa: isize, rsb: isize, csb: isize) {
        if std::env::var("VNN_RECURSIVE_MATMUL").is_ok() {
            // Experimental Cache-Oblivious path
            out.fill(0.0);
            crate::tiling_cpu::matmul_recursive(a, b, out, m, k, n, rsa as usize, csa as usize, rsb as usize, csb as usize, n, 1);
            return;
        }

        let next_m = std::sync::atomic::AtomicUsize::new(0);
        let ptr_out = out.as_mut_ptr() as usize; let ptr_a = a.as_ptr() as usize; let ptr_b = b.as_ptr() as usize;
        let blk = 512;
        rayon::scope(|s| {
            for _ in 0..rayon::current_num_threads() {
                s.spawn(|_| {
                    loop {
                        let ms = next_m.fetch_add(blk, std::sync::atomic::Ordering::Relaxed);
                        if ms >= m { break; }
                        let cur_m = (m - ms).min(blk);
                        unsafe { 
                            matrixmultiply::sgemm(cur_m, k, n, 1.0, 
                                  (ptr_a as *const f32).offset(ms as isize * rsa), rsa, csa, 
                                  ptr_b as *const f32, rsb, csb, 
                                  0.0, (ptr_out as *mut f32).add(ms * n), n as isize, 1); 
                        }
                    }
                });
            }
        });
    }

    fn cpu_sgemm_streamed_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize, rsa: isize, csa: isize, rsb: isize, csb: isize) {
        let blk_m = 512; let blk_k = 2048;
        for ks in (0..k).step_by(blk_k) {
            let bk = (k - ks).min(blk_k);
            let mut b_band = vec![0.0; bk * n];
            b_band.par_chunks_mut(n).enumerate().for_each(|(i, r)| {
                let k_idx = ks + i;
                for j in 0..n {
                    let idx = k_idx as isize * rsb + j as isize * csb;
                    r[j] = b[idx as usize];
                }
            });
            let next_m = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let ptr_out = out.as_mut_ptr() as usize;
            // Bug #6 fix: Pre-allocate a_band buffers OUTSIDE the rayon scope to avoid
            // per-iteration heap allocation inside work-stealing threads. Each thread gets
            // its own pre-sized buffer; we use a Vec<Vec<f32>> indexed by thread position.
            let num_threads = rayon::current_num_threads();
            let mut a_bands: Vec<Vec<f32>> = (0..num_threads).map(|_| vec![0.0f32; blk_m * bk]).collect();
            // Pass b_band as raw ptr (usize) so it can be shared across closures without move.
            let b_band_ptr = b_band.as_ptr() as usize;
            rayon::scope(|s| {
                for t in 0..num_threads {
                    // Safety: each thread writes only to its own a_bands[t] slot.
                    let a_band_ptr = a_bands[t].as_mut_ptr() as usize;
                    let next_m_t = next_m.clone();
                    s.spawn(move |_| {
                        loop {
                            let ms = next_m_t.fetch_add(blk_m, std::sync::atomic::Ordering::Relaxed);
                            if ms >= m { break; }
                            let cur_m = (m - ms).min(blk_m);
                            // Reuse pre-allocated a_band buffer (Bug #6 fix)
                            let a_band = unsafe { std::slice::from_raw_parts_mut(a_band_ptr as *mut f32, cur_m * bk) };
                            for i in 0..cur_m {
                                let m_idx = ms + i;
                                for j in 0..bk {
                                    let k_idx = ks + j;
                                    let idx = m_idx as isize * rsa + k_idx as isize * csa;
                                    a_band[i * bk + j] = a[idx as usize];
                                }
                            }
                            unsafe { matrixmultiply::sgemm(cur_m, bk, n, 1.0, a_band.as_ptr(), bk as isize, 1, b_band_ptr as *const f32, n as isize, 1, if ks == 0 { 0.0 } else { 1.0 }, (ptr_out as *mut f32).add(ms * n), n as isize, 1); }
                        }
                    });
                }
            });
        }
    }
}
