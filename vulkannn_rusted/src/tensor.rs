use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, ToPyArray};
use numpy::ndarray::Array;
use rayon::prelude::*;
use half::f16;

#[derive(Clone)]
pub enum MmapType {
    ReadOnly(std::sync::Arc<memmap2::Mmap>),
    ReadWrite(std::sync::Arc<memmap2::MmapMut>),
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[pyclass(eq, eq_int)]
pub enum DataType {
    F32,
    F16,
}

#[derive(Clone)]
pub enum Storage {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    None,
}

#[pyclass(unsendable)]
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
    pub mmap_data: Option<MmapType>,
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
                    let f16_vec = vec.par_iter().map(|&x| half::f16::from_f32(x)).collect();
                    (Storage::F16(f16_vec), nd_arr.shape().to_vec())
                },
            };
            (storage, final_shape)
        } else if let Some(s) = shape {
            let size = s.iter().product();
            match dtype {
                DataType::F32 => (Storage::F32(vec![0.0; size]), s),
                DataType::F16 => (Storage::F16(vec![half::f16::ZERO; size]), s),
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
        let bytes_per_elem = match dtype { DataType::F32 => 4, DataType::F16 => 2 };
        let expected_size = shape.iter().product::<usize>() * bytes_per_elem;
        if metadata.len() < expected_size as u64 {
            return Err(PyValueError::new_err(format!("File size mismatch: expected at least {} bytes, found {}", expected_size, metadata.len())));
        }
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).map_err(|e| PyValueError::new_err(e.to_string()))? };
        #[cfg(target_os = "linux")] unsafe { libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_SEQUENTIAL); }
        Ok(Tensor { shape, device: "ssd".to_string(), name: "SSDMapped".to_string(), is_transposed: false, dtype, storage: Storage::None, mmap_data: Some(MmapType::ReadOnly(std::sync::Arc::new(mmap))) })
    }

    #[staticmethod]
    #[pyo3(signature = (path, shape, dtype=DataType::F32))]
    fn new_ssd(path: &str, shape: Vec<usize>, dtype: DataType) -> PyResult<Self> {
        let size = shape.iter().product::<usize>();
        let bytes_per_elem = match dtype { DataType::F32 => 4, DataType::F16 => 2 };
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        file.set_len((size * bytes_per_elem) as u64).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file).map_err(|e| PyValueError::new_err(e.to_string()))? };
        #[cfg(target_os = "linux")] unsafe { libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_SEQUENTIAL); }
        Ok(Tensor { shape, device: "ssd".to_string(), name: "SSDResult".to_string(), is_transposed: false, dtype, storage: Storage::None, mmap_data: Some(MmapType::ReadWrite(std::sync::Arc::new(mmap))) })
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        let vec = match self.dtype {
            DataType::F32 => {
                let (slice, _) = self.get_slice_raw_f32();
                slice.to_vec()
            },
            DataType::F16 => {
                let (slice, _) = self.get_slice_raw_f16();
                slice.par_iter().map(|&x| f32::from(x)).collect()
            }
        };
        Ok(Array::from_shape_vec(self.shape.clone(), vec).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray_bound(py))
    }

    fn to_numpy_no_copy<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, numpy::PyArrayDyn<f32>>, bool)> {
        match self.dtype {
            DataType::F32 => {
                let (slice, is_ssd) = self.get_slice_raw_f32();
                let shape = self.shape.clone();
                let array = unsafe { numpy::ndarray::ArrayViewD::from_shape_ptr(shape, slice.as_ptr()) };
                Ok((array.to_pyarray_bound(py), is_ssd))
            },
            DataType::F16 => {
                // Cannot do no-copy view of F16 as F32. Fallback to copy.
                let arr = self.to_numpy(py)?;
                Ok((arr, false))
            }
        }
    }

    fn __repr__(&self) -> String { format!("Tensor(shape={:?}, dtype={:?}, device='{}', ssd={})", self.shape, self.dtype, self.device, self.mmap_data.is_some()) }

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
            } else {
                let (a, _) = self.get_slice_raw_f16();
                let (b, _) = other.get_slice_raw_f16();
                let mut res = vec![f16::ZERO; a.len()];
                res.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &a), &b)| *c = f16::from_f32(a.to_f32() + b.to_f32()));
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: "AddRes".to_string(), is_transposed: false, dtype: DataType::F16, storage: Storage::F16(res), mmap_data: None })
            }
        } else {
            let (a_raw, _) = self.get_slice_raw_bytes();
            let (b_raw, _) = other.get_slice_raw_bytes();
            let res_raw = crate::backend::execute_add(a_raw, b_raw, self.dtype, self.device == "hybrid");
            let storage = if self.dtype == DataType::F16 {
                Storage::F16(bytemuck::cast_slice(&res_raw).to_vec())
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
            }
        } else {
            let (a, _) = self.get_slice_raw_bytes();
            let (b, _) = other.get_slice_raw_bytes();
            let (out_raw, _) = out.get_slice_raw_mut_bytes();
            crate::backend::execute_add_into(a, b, out_raw, self.dtype, self.device == "hybrid", false);
            Ok(())
        }
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
            let bytes_per_elem = match dtype { DataType::F32 => 4, DataType::F16 => 2 };
            let sz = (m*k + k*n + m*n) * bytes_per_elem;

            let storage = if sz < safe_ram {
                match dtype {
                    DataType::F32 => {
                        let mut out = vec![0.0; m * n];
                        let (a, a_ssd) = self.get_slice_raw_f32();
                        let (b, b_ssd) = other.get_slice_raw_f32();
                        if a_ssd || b_ssd {
                            let ar = if a_ssd { self.par_copy_f32(a) } else { a.to_vec() };
                            let br = if b_ssd { self.par_copy_f32(b) } else { b.to_vec() };
                            self.cpu_sgemm_f32(&ar, &br, &mut out, m, k, n, rsa, csa, rsb, csb);
                        } else {
                            self.cpu_sgemm_f32(a, b, &mut out, m, k, n, rsa, csa, rsb, csb);
                        }
                        Storage::F32(out)
                    },
                        let mut out = vec![half::f16::ZERO; m * n];
                        let (a, a_ssd) = self.get_slice_raw_f16();
                        let (b, b_ssd) = other.get_slice_raw_f16();
                        let a_f32: Vec<f32> = a.par_iter().map(|&x| f32::from(x)).collect();
                        let b_f32: Vec<f32> = b.par_iter().map(|&x| f32::from(x)).collect();
                        let mut res_f32 = vec![0.0; m * n];
                        
                        self.cpu_sgemm_f32(&a_f32, &b_f32, &mut res_f32, m, k, n, rsa, csa, rsb, csb);
                        
                        out.par_iter_mut().zip(res_f32.par_iter()).for_each(|(o, &x)| *o = half::f16::from_f32(x));
                        Storage::F16(out)
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
                    DataType::F16 => {
                        let mut out = vec![half::f16::ZERO; m * n];
                        let (a, _) = self.get_slice_raw_f16();
                        let (b, _) = other.get_slice_raw_f16();
                        unsafe {
                            gemm::gemm::<gemm::f16>(
                                m, n, k,
                                out.as_mut_ptr() as *mut gemm::f16, n as isize, 1,
                                false,
                                a.as_ptr() as *const gemm::f16, rsa, csa,
                                b.as_ptr() as *const gemm::f16, rsb, csb,
                                gemm::f16::from_f32(1.0), gemm::f16::from_f32(0.0),
                                false, false, false,
                                gemm::Parallelism::Rayon(rayon::current_num_threads()),
                            );
                        }
                        Storage::F16(out)
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
    fn check_shape(&self, other: &Tensor) -> PyResult<()> {
        if self.shape != other.shape { return Err(PyValueError::new_err("Shape mismatch")); }
        if self.dtype != other.dtype { return Err(PyValueError::new_err("DType mismatch")); }
        Ok(())
    }

    fn unary_op(&self, op: &str) -> PyResult<Tensor> {
        if self.device == "cpu" {
            if self.dtype == DataType::F32 {
                let (input, _) = self.get_slice_raw_f32();
                let mut res = vec![0.0; input.len()];
                Self::act_into_raw_f32(input, op, &mut res)?;
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::F32, storage: Storage::F32(res), mmap_data: None })
            } else {
                let (input, _) = self.get_slice_raw_f16();
                let mut res = vec![f16::ZERO; input.len()];
                match op {
                    "relu" => res.par_iter_mut().zip(input.par_iter()).for_each(|(o, &i)| *o = if i.to_f32() > 0.0 { i } else { f16::ZERO }),
                    _ => panic!("Unsupported F16 CPU op: {}", op),
                }
                Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: DataType::F16, storage: Storage::F16(res), mmap_data: None })
            }
        } else {
            let (input_raw, _) = self.get_slice_raw_bytes();
            let res_raw = crate::backend::execute_activation(input_raw, op, self.dtype, self.device == "hybrid");
            let storage = if self.dtype == DataType::F16 {
                Storage::F16(bytemuck::cast_slice(&res_raw).to_vec())
            } else {
                Storage::F32(bytemuck::cast_slice(&res_raw).to_vec())
            };
            Ok(Tensor { shape: self.shape.clone(), device: self.device.clone(), name: format!("{}_res", op), is_transposed: false, dtype: self.dtype, storage, mmap_data: None })
        }
    }

    pub fn get_slice_raw_bytes(&self) -> (&[u8], bool) {
        match &self.storage {
            Storage::F32(v) => (bytemuck::cast_slice(v), false),
            Storage::F16(v) => (bytemuck::cast_slice(v), false),
            Storage::None => {
                if let Some(m) = &self.mmap_data {
                    match m {
                        MmapType::ReadOnly(inner) => (&inner[..], true),
                        MmapType::ReadWrite(inner) => (&inner[..], true),
                    }
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
                if let Some(mmap) = &self.mmap_data {
                    match mmap {
                        MmapType::ReadOnly(m) => (unsafe { std::slice::from_raw_parts(m.as_ptr() as *const f32, m.len()/4) }, true),
                        MmapType::ReadWrite(m) => (unsafe { std::slice::from_raw_parts(m.as_ptr() as *const f32, m.len()/4) }, true),
                    }
                } else { (&[], false) }
            },
            _ => (&[], false),
        }
    }

    fn get_slice_raw_f16(&self) -> (&[half::f16], bool) {
        match &self.storage {
            Storage::F16(v) => (v, false),
            Storage::None => {
                if let Some(mmap) = &self.mmap_data {
                    match mmap {
                        MmapType::ReadOnly(m) => (unsafe { std::slice::from_raw_parts(m.as_ptr() as *const half::f16, m.len()/2) }, true),
                        MmapType::ReadWrite(m) => (unsafe { std::slice::from_raw_parts(m.as_ptr() as *const half::f16, m.len()/2) }, true),
                    }
                } else { (&[], false) }
            },
            _ => (&[], false),
        }
    }

    pub fn get_slice_raw_mut_bytes(&mut self) -> (&mut [u8], bool) {
        match &mut self.storage {
            Storage::F32(v) => (bytemuck::cast_slice_mut(v), false),
            Storage::F16(v) => (bytemuck::cast_slice_mut(v), false),
            Storage::None => {
                if let Some(mmap) = &mut self.mmap_data {
                    match mmap {
                        MmapType::ReadWrite(m) => (unsafe { std::slice::from_raw_parts_mut(m.as_ptr() as *mut u8, m.len()) }, true),
                        MmapType::ReadOnly(_) => panic!("Modify ReadOnly SSD!"),
                    }
                } else { panic!("No data") }
            },
        }
    }

    fn get_slice_raw_mut_f32(&mut self) -> (&mut [f32], bool) {
        match &mut self.storage {
            Storage::F32(v) => (v, false),
            Storage::None => {
                if let Some(mmap) = &self.mmap_data {
                    match mmap {
                        MmapType::ReadWrite(m) => (unsafe { std::slice::from_raw_parts_mut(m.as_ptr() as *mut f32, m.len()/4) }, true),
                        MmapType::ReadOnly(_) => panic!("Modify ReadOnly SSD!"),
                    }
                } else { panic!("No data") }
            },
            _ => panic!("Wrong DType"),
        }
    }

    fn get_slice_raw_mut_f16(&mut self) -> (&mut [half::f16], bool) {
        match &mut self.storage {
            Storage::F16(v) => (v, false),
            Storage::None => {
                if let Some(mmap) = &self.mmap_data {
                    match mmap {
                        MmapType::ReadWrite(m) => (unsafe { std::slice::from_raw_parts_mut(m.as_ptr() as *mut half::f16, m.len()/2) }, true),
                        MmapType::ReadOnly(_) => panic!("Modify ReadOnly SSD!"),
                    }
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
        if total_elements < 1_000_000 {
                match op {
                    "relu" => out.par_iter_mut().zip(i_s.par_iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 }),
                    "sigmoid" => out.par_iter_mut().zip(i_s.par_iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp())),
                    "silu" => out.par_iter_mut().zip(i_s.par_iter()).for_each(|(o, &i)| *o = i * (1.0 / (1.0 + (-i).exp()))),
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
                            "relu" => for k in 0..process_len { chunk[k] = if isub[k] > 0.0 { isub[k] } else { 0.0 }; },
                            "sigmoid" => for k in 0..process_len { chunk[k] = 1.0 / (1.0 + (-isub[k]).exp()); },
                            "silu" => for k in 0..process_len { chunk[k] = isub[k] * (1.0 / (1.0 + (-isub[k]).exp())); },
                            _ => {}
                        }
                    }
                });
            }
        Ok(())
    }

    fn par_copy_f32(&self, src: &[f32]) -> Vec<f32> {
        let mut dst = vec![0.0; src.len()];
        dst.par_chunks_mut(1024*1024).enumerate().for_each(|(i, c)| {
            let s = i * 1024 * 1024;
            let end = std::cmp::min(s + c.len(), src.len());
            if s < end { c.copy_from_slice(&src[s..end]); }
        });
        dst
    }

    fn par_copy_f16(&self, src: &[half::f16]) -> Vec<half::f16> {
        let mut dst = vec![half::f16::ZERO; src.len()];
        dst.par_chunks_mut(1024*1024).enumerate().for_each(|(i, c)| {
            let s = i * 1024 * 1024;
            let end = std::cmp::min(s + c.len(), src.len());
            if s < end { c.copy_from_slice(&src[s..end]); }
        });
        dst
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
            let next_m = std::sync::atomic::AtomicUsize::new(0);
            let ptr_out = out.as_mut_ptr() as usize;
            rayon::scope(|s| {
                for _ in 0..rayon::current_num_threads() {
                    s.spawn(|_| {
                        loop {
                            let ms = next_m.fetch_add(blk_m, std::sync::atomic::Ordering::Relaxed);
                            if ms >= m { break; }
                            let cur_m = (m - ms).min(blk_m);
                            let mut a_band = vec![0.0; cur_m * bk];
                            for i in 0..cur_m {
                                let m_idx = ms + i;
                                for j in 0..bk {
                                    let k_idx = ks + j;
                                    let idx = m_idx as isize * rsa + k_idx as isize * csa;
                                    a_band[i * bk + j] = a[idx as usize];
                                }
                            }
                            unsafe { matrixmultiply::sgemm(cur_m, bk, n, 1.0, a_band.as_ptr(), bk as isize, 1, b_band.as_ptr(), n as isize, 1, if ks == 0 { 0.0 } else { 1.0 }, (ptr_out as *mut f32).add(ms * n), n as isize, 1); }
                        }
                    });
                }
            });
        }
    }
}
