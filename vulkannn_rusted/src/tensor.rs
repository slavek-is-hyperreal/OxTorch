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
}

#[derive(Clone)]
pub enum Storage {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    BF16(Vec<half::bf16>),
    None,
}

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
                }
            };
            (storage, final_shape)
        } else if let Some(s) = shape {
            let size = s.iter().product();
            match dtype {
                DataType::F32 => (Storage::F32(vec![0.0; size]), s),
                DataType::F16 => (Storage::F16(vec![half::f16::ZERO; size]), s),
                DataType::BF16 => (Storage::BF16(vec![half::bf16::ZERO; size]), s),
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
        let bytes_per_elem = if dtype == DataType::F32 { 4 } else { 2 };
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
        let bytes_per_elem = if dtype == DataType::F32 { 4 } else { 2 };
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        file.set_len((size * bytes_per_elem) as u64).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let engine = DirectIoEngine::new(path, false);
        Ok(Tensor { shape, device: "ssd".to_string(), name: "SSDResult".to_string(), is_transposed: false, dtype, storage: Storage::None, mmap_data: Some(IoEngineType::ReadWrite(Arc::new(engine))) })
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
            }
        };
        Ok(Array::from_shape_vec(self.shape.clone(), vec).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray_bound(py))
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
            DataType::F16 | DataType::BF16 => {
                // Cannot do no-copy view of F16/BF16 as F32. Fallback to copy.
                let arr = self.to_numpy(py)?;
                Ok((arr, false))
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
                    let (a, _) = self.get_slice_raw_bf16();
                    let (b, _) = other.get_slice_raw_bf16();
                    out_slice.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &av), &bv)| *c = half::bf16::from_f32(av.to_f32() + bv.to_f32()));
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

    fn relu(&self) -> PyResult<Tensor> { self.unary_op("relu") }
    fn relu_into(&mut self, out: &mut Tensor) -> PyResult<()> { self.act_into("relu", out) }
    fn sigmoid_into(&mut self, out: &mut Tensor) -> PyResult<()> { self.act_into("sigmoid", out) }
    fn silu_into(&mut self, out: &mut Tensor) -> PyResult<()> { self.act_into("silu", out) }

    fn act_into(&mut self, op: &str, out: &mut Tensor) -> PyResult<()> {
        if self.device == "cpu" {
            match self.dtype {
                DataType::F32 => {
                    let (out_slice, _) = out.get_slice_raw_mut_f32();
                    let (i_s, _) = self.get_slice_raw_f32();
                    Self::act_into_raw_f32(i_s, op, out_slice)
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
                    match op {
                        "relu" => out_slice.par_iter_mut().zip(i_s.par_iter()).for_each(|(o, &i)| *o = if i > half::bf16::ZERO { i } else { half::bf16::ZERO }),
                        _ => panic!("Unsupported BF16 CPU op into: {}", op),
                    }
                    Ok(())
                }
            }
        } else {
            let (input_raw, _) = self.get_slice_raw_bytes();
            let (res_raw, _) = out.get_slice_raw_mut_bytes();
            crate::backend::execute_activation_into(input_raw, op, res_raw, self.dtype, self.device == "hybrid", false);
            Ok(())
        }
    }

    fn sigmoid(&self) -> PyResult<Tensor> { self.unary_op("sigmoid") }
    fn silu(&self) -> PyResult<Tensor> { self.unary_op("silu") }

    // --- MATMUL ---
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
            let bytes_per_elem = if dtype == DataType::F32 { 4 } else { 2 };
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
                    DataType::F16 | DataType::BF16 => {
                        let is_f16 = dtype == DataType::F16;
                        let mut out_bytes = vec![0u8; m * n * 2];
                        let (a_bytes, _) = self.get_slice_raw_bytes();
                        let (b_bytes, _) = other.get_slice_raw_bytes();
                        
                        // Streamed F16/BF16: For now fallback to F32 via streaming then convert back
                        // Optimizing this for streaming BF16 next.
                        let mut res_f32 = BufPool::get(m * n);
                        res_f32.resize(m * n, 0.0f32);
                        let (a_f32, b_f32) = if is_f16 {
                            let a_f16: &[half::f16] = bytemuck::cast_slice(a_bytes);
                            let b_f16: &[half::f16] = bytemuck::cast_slice(b_bytes);
                            (a_f16.par_iter().map(|&x| x.to_f32()).collect::<Vec<_>>(),
                             b_f16.par_iter().map(|&x| x.to_f32()).collect::<Vec<_>>())
                        } else {
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
                        } else {
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
            } else {
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
            }
        } else {
            let (a_raw, _) = self.get_slice_raw_bytes();
            let (b_raw, _) = other.get_slice_raw_bytes();
            let res_raw = crate::backend::execute_elementwise(a_raw, b_raw, op, self.dtype, self.device == "hybrid");
            let storage = if self.dtype == DataType::F16 {
                Storage::F16(bytemuck::cast_slice(&res_raw).to_vec())
            } else if self.dtype == DataType::BF16 {
                Storage::BF16(bytemuck::cast_slice(&res_raw).to_vec())
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
            } else {
                let (input, _) = self.get_slice_raw_bf16();
                let mut res = vec![bf16::ZERO; input.len()];
                match op {
                    "mul_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() * val)),
                    "add_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() + val)),
                    "sub_scalar" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() - val)),
                    "div_scalar" => { let inv = 1.0 / val; res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = bf16::from_f32(i.to_f32() * inv)); },
                    _ => {}
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "ScalarRes".to_string(), is_transposed: false, dtype: self.dtype, storage: Storage::BF16(res), mmap_data: None })
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

    fn unary_op(&self, op: &str) -> PyResult<Tensor> {
        if self.is_ssd() {
            return self.unary_op_ssd(op);
        }
        if self.device == "cpu" {
            if self.dtype == DataType::F32 {
                let (input, _) = self.get_slice_raw_f32();
                // BufPool::get reuses warm memory from a previous call — avoids cold OS page fault.
                // This is the primary fix for relu() being 10x slower than PyTorch on small tensors.
                let mut res = BufPool::get(input.len());
                Self::act_into_raw_f32(input, op, &mut res)?;
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(res), mmap_data: None })
            } else if self.dtype == DataType::F16 {
                let (input, _) = self.get_slice_raw_f16();
                let mut res = vec![f16::ZERO; input.len()];
                match op {
                    "relu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = if i.to_f32() > 0.0 { i } else { f16::ZERO }),
                    _ => panic!("Unsupported F16 CPU op: {}", op),
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::F16, storage: Storage::F16(res), mmap_data: None })
            } else if self.dtype == DataType::BF16 {
                let (input, _) = self.get_slice_raw_bf16();
                let mut res = vec![bf16::ZERO; input.len()];
                match op {
                    "relu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = if i.to_f32() > 0.0 { i } else { bf16::ZERO }),
                    _ => panic!("Unsupported BF16 CPU op: {}", op),
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::BF16, storage: Storage::BF16(res), mmap_data: None })
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
            let num_total = input_raw.len() / (if self.dtype == DataType::F32 { 4 } else { 2 });

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
                        let bpe = if dtype == DataType::F32 { 4usize } else { 2usize };
                        // Safety: raw ptrs are valid for `total_len` bytes
                        let in_sl  = unsafe { std::slice::from_raw_parts    (input_ptr  as *const u8, total_len) };
                        let out_sl = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut   u8, total_len) };
                        let _ = bpe; // suppress lint
                        crate::backend::execute_activation_chunked(in_sl, out_sl, elem_start, elem_count, &op_gpu, dtype);
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
                        let bpe = if dtype == DataType::F32 { 4usize } else { 2usize };
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
                                    _ => panic!("Unsupported CPU op in hybrid tile worker"),
                                }
                            },
                            DataType::F16 => {
                                let in_h  = unsafe { std::slice::from_raw_parts    (in_sl.as_ptr()  .add(elem_start * bpe) as *const half::f16, elem_count) };
                                let out_h = unsafe { std::slice::from_raw_parts_mut(out_sl.as_mut_ptr().add(elem_start * bpe) as *mut half::f16, elem_count) };
                                match op_cpu.as_str() {
                                    "relu" => out_h.iter_mut().zip(in_h.iter()).for_each(|(o, &i)| *o = if i > half::f16::ZERO { i } else { half::f16::ZERO }),
                                    _ => panic!("Unsupported F16 hybrid tile op"),
                                }
                            },
                            DataType::BF16 => {
                                let in_b  = unsafe { std::slice::from_raw_parts    (in_sl.as_ptr()  .add(elem_start * bpe) as *const half::bf16, elem_count) };
                                let out_b = unsafe { std::slice::from_raw_parts_mut(out_sl.as_mut_ptr().add(elem_start * bpe) as *mut half::bf16, elem_count) };
                                match op_cpu.as_str() {
                                    "relu" => out_b.iter_mut().zip(in_b.iter()).for_each(|(o, &i)| *o = if i > half::bf16::ZERO { i } else { half::bf16::ZERO }),
                                    _ => panic!("Unsupported BF16 hybrid tile op"),
                                }
                            },
                        }
                    }
                });
                } // end for cpu workers
            });

            let storage = if dtype == DataType::F16 {
                Storage::F16(bytemuck::cast_slice(&out_raw).to_vec())
            } else if dtype == DataType::BF16 {
                Storage::BF16(bytemuck::cast_slice(&out_raw).to_vec())
            } else {
                Storage::F32(bytemuck::cast_slice(&out_raw).to_vec())
            };
            Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: self.dtype, storage, mmap_data: None })

        } else {
            // Pure Vulkan path (device == "vulkan")
            let (input_raw, _) = self.get_slice_raw_bytes();
            let res_raw = crate::backend::execute_activation(input_raw, op, self.dtype, false);
            let storage = if self.dtype == DataType::F16 {
                Storage::F16(bytemuck::cast_slice(&res_raw).to_vec())
            } else if self.dtype == DataType::BF16 {
                Storage::BF16(bytemuck::cast_slice(&res_raw).to_vec())
            } else {
                Storage::F32(bytemuck::cast_slice(&res_raw).to_vec())
            };
            Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: self.dtype, storage, mmap_data: None })
        }
    }

    fn unary_op_ssd(&self, op: &str) -> PyResult<Tensor> {
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
        
        let total_elements = self.shape.iter().product::<usize>();
        let bytes_per_elem = if self.dtype == DataType::F32 { 4 } else { 2 };
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
                    Self::act_into_raw_parallel_f32(slice, op);
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

    fn act_into_raw_parallel_f32(slice: &mut [f32], op: &str) {
        match op {
            "relu" => slice.par_iter_mut().for_each(|x| if *x < 0.0 { *x = 0.0; }),
            "sigmoid" => slice.par_iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp())),
            "silu" => slice.par_iter_mut().for_each(|x| *x = *x / (1.0 + (-*x).exp())),
            _ => panic!("Unsupported Op: {}", op),
        }
    }

    pub fn get_slice_raw_bytes(&self) -> (&[u8], bool) {
        match &self.storage {
            Storage::F32(v) => (bytemuck::cast_slice(v), false),
            Storage::F16(v) => (bytemuck::cast_slice(v), false),
            Storage::BF16(v) => (bytemuck::cast_slice(v), false),
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

    pub fn get_slice_raw_mut_bytes(&mut self) -> (&mut [u8], bool) {
        match &mut self.storage {
            Storage::F32(v) => (bytemuck::cast_slice_mut(v), false),
            Storage::F16(v) => (bytemuck::cast_slice_mut(v), false),
            Storage::BF16(v) => (bytemuck::cast_slice_mut(v), false),
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

    fn act_into_raw_f32(i_s: &[f32], op: &str, out: &mut [f32]) -> PyResult<()> {
        let total_elements = i_s.len();
        // Bug #4 fix: was threshold 1_000_000 (4MB). Rayon spawn overhead dominates for simple
        // element-wise ops below ~32MB. Per Rayon docs: use serial iter for small workloads.
        // PyTorch uses a single-threaded kernel for small tensors — same principle here.
        // 8_000_000 floats = 32MB, well above L3 cache, good break-even for thread overhead.
        if total_elements < 8_000_000 {
            // Serial path — AVX1 vmaxps (8 floats/cycle) for relu, scalar fallback for rest
            match op {
                "relu"    => crate::avx_swar::relu_f32(i_s, out),
                "sigmoid" => { for (o, &i) in out.iter_mut().zip(i_s.iter()) { *o = 1.0 / (1.0 + (-i).exp()); } },
                "silu"    => { for (o, &i) in out.iter_mut().zip(i_s.iter()) { *o = i * (1.0 / (1.0 + (-i).exp())); } },
                _ =>{}
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
        let bytes_per_elem = if self.dtype == DataType::F32 { 4 } else { 2 };
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
