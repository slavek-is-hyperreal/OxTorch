use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use numpy::ndarray::Array;
use rayon::prelude::*;
use matrixmultiply::sgemm;

/// The central Tensor struct mimicking `vulkan_nn_lib.tensor.Tensor`.
#[pyclass(unsendable)]
pub struct Tensor {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get, set)]
    pub device: String,
    #[pyo3(get, set)]
    pub name: String,
    
    // Internal CPU representation for fallback and initialization
    pub cpu_data: Vec<f32>,
    
    // Zero-copy SSD memmap
    pub mmap_data: Option<std::sync::Arc<memmap2::Mmap>>,
}

#[pymethods]
impl Tensor {
    /// Initialize a new Tensor.
    #[new]
    #[pyo3(signature = (data=None, shape=None, dtype=None, device="auto", name="Tensor"))]
    #[allow(unused_variables)]
    fn new(
        data: Option<PyReadonlyArrayDyn<'_, f32>>,
        shape: Option<Vec<usize>>,
        dtype: Option<Bound<'_, PyAny>>,
        device: &str,
        name: &str,
    ) -> PyResult<Self> {
        let (cpu_data, final_shape) = if let Some(arr) = data {
            let nd_arr = arr.as_array();
            let slice = nd_arr.as_slice().unwrap_or(&[]).to_vec();
            (slice, nd_arr.shape().to_vec())
        } else if let Some(s) = shape {
            let total_size: usize = s.iter().product();
            (vec![0.0; total_size], s)
        } else {
            return Err(PyValueError::new_err("Must provide either data or shape"));
        };

        Ok(Tensor {
            shape: final_shape,
            device: device.to_string(),
            name: name.to_string(),
            cpu_data,
            mmap_data: None,
        })
    }

    #[staticmethod]
    fn from_ssd(path: &str, shape: Vec<usize>) -> PyResult<Self> {
        let mmap = crate::streaming::L3Cache::map_ssd_tensor(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        crate::streaming::prefetch_tensor(mmap.clone());
        Ok(Tensor {
            shape,
            device: "ssd".to_string(),
            name: "SSDMapped".to_string(),
            cpu_data: Vec::new(),
            mmap_data: Some(mmap),
        })
    }

    #[staticmethod]
    fn new_ssd(path: &str, shape: Vec<usize>) -> PyResult<Self> {
        let total_size = shape.iter().product::<usize>();
        let file = std::fs::OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to create SSD file: {}", e)))?;
        file.set_len((total_size * 4) as u64).map_err(|e| PyValueError::new_err(format!("Failed to set SSD file length: {}", e)))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|e| PyValueError::new_err(format!("Failed to mmap SSD file: {}", e)))?;
        Ok(Tensor {
            shape,
            device: "ssd".to_string(),
            name: "SSDResult".to_string(),
            cpu_data: Vec::new(),
            mmap_data: Some(std::sync::Arc::new(mmap)),
        })
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        let (slice, _) = self.get_slice_raw();
        let array = Array::from_shape_vec(self.shape.clone(), slice.to_vec())
            .map_err(|e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string()))?;
        Ok(array.into_pyarray_bound(py))
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        self.numpy(py)
    }

    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, device='{}', name='{}')", self.shape, self.device, self.name)
    }

    fn __add__(&self, other: &Tensor) -> PyResult<Tensor> {
        if self.shape != other.shape { return Err(PyValueError::new_err("Shapes must match")); }
        let (a_slice, _) = self.get_slice_raw();
        let mut result_data = vec![0.0; a_slice.len()];
        self.add_into_raw(other, &mut result_data)?;
        Ok(Tensor {
            shape: self.shape.clone(),
            device: self.device.clone(),
            name: format!("({}+{})", self.name, other.name),
            cpu_data: result_data,
            mmap_data: None,
        })
    }

    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(PyValueError::new_err("Tensors must be 2D for basic MatMul."));
        }
        let m = self.shape[0] as u32;
        let k = self.shape[1] as u32;
        let k2 = other.shape[0] as u32;
        let n = other.shape[1] as u32;
        if k != k2 { return Err(PyValueError::new_err(format!("Shape mismatch: {:?} x {:?}", self.shape, other.shape))); }

        let (a_slice, _) = self.get_slice_raw();
        let (b_slice, _) = other.get_slice_raw();

        let result_data = if self.device == "cpu" {
            let mut out = vec![0.0; (m * n) as usize];
            crate::streaming::init_budgets();
            let budgets = crate::streaming::BUDGETS.get().unwrap().lock().unwrap();
            let ram_limit = budgets.l2_ram_max_bytes;
            
            let size_a = (m as usize * k as usize) * 4;
            let size_b = (k as usize * n as usize) * 4;
            let size_c = (m as usize * n as usize) * 4;

            let par_copy = |src: &[f32]| -> Vec<f32> {
                let mut dst = vec![0.0; src.len()];
                let chunk_size = 1024 * 1024;
                dst.par_chunks_mut(chunk_size).enumerate().for_each(|(i, chunk)| {
                    let start = i * chunk_size;
                    let end = std::cmp::min(start + chunk_size, src.len());
                    chunk.copy_from_slice(&src[start..end]);
                });
                dst
            };

            if size_a + size_b + size_c < ram_limit {
                let (a_in_ram, b_in_ram) = rayon::join(|| par_copy(a_slice), || par_copy(b_slice));
                let next_m = std::sync::atomic::AtomicU32::new(0);
                let out_ptr = out.as_mut_ptr() as usize;
                let a_ptr = a_in_ram.as_ptr() as usize;
                let b_ptr = b_in_ram.as_ptr() as usize;
                let block_m = 512;
                rayon::scope(|s| {
                    for _ in 0..rayon::current_num_threads() {
                        s.spawn(|_| {
                            loop {
                                let m_start = next_m.fetch_add(block_m as u32, std::sync::atomic::Ordering::Relaxed) as usize;
                                if m_start >= m as usize { break; }
                                let bm = std::cmp::min(block_m, m as usize - m_start);
                                unsafe { sgemm(bm, k as usize, n as usize, 1.0, (a_ptr as *const f32).add(m_start * k as usize), k as isize, 1, b_ptr as *const f32, n as isize, 1, 0.0, (out_ptr as *mut f32).add(m_start * n as usize), n as isize, 1); }
                            }
                        });
                    }
                });
            } else if size_b + size_c < ram_limit {
                let b_in_ram = par_copy(b_slice);
                let next_m = std::sync::atomic::AtomicU32::new(0);
                let out_ptr = out.as_mut_ptr() as usize;
                let b_ptr = b_in_ram.as_ptr() as usize;
                let block_m = 512;
                rayon::scope(|s| {
                    for _ in 0..rayon::current_num_threads() {
                        s.spawn(|_| {
                            loop {
                                let m_start = next_m.fetch_add(block_m as u32, std::sync::atomic::Ordering::Relaxed) as usize;
                                if m_start >= m as usize { break; }
                                let bm = std::cmp::min(block_m, m as usize - m_start);
                                let mut a_band = vec![0.0; bm * k as usize];
                                a_band.copy_from_slice(&a_slice[m_start * k as usize .. (m_start + bm) * k as usize]);
                                unsafe { sgemm(bm, k as usize, n as usize, 1.0, a_band.as_ptr(), k as isize, 1, b_ptr as *const f32, n as isize, 1, 0.0, (out_ptr as *mut f32).add(m_start * n as usize), n as isize, 1); }
                            }
                        });
                    }
                });
            } else {
                let block_m = 512; let block_k = 2048;
                for k_start in (0..k as usize).step_by(block_k) {
                    let bk = std::cmp::min(block_k, k as usize - k_start);
                    let mut b_band = vec![0.0; bk * n as usize];
                    let b_src = &b_slice[k_start * n as usize .. (k_start + bk) * n as usize];
                    b_band.par_chunks_mut(n as usize).enumerate().for_each(|(i, row)| { row.copy_from_slice(&b_src[i * n as usize .. (i + 1) * n as usize]); });
                    let next_m = std::sync::atomic::AtomicU32::new(0);
                    let out_ptr = out.as_mut_ptr() as usize;
                    rayon::scope(|s| {
                        for _ in 0..rayon::current_num_threads() {
                            s.spawn(|_| {
                                loop {
                                    let m_start = next_m.fetch_add(block_m as u32, std::sync::atomic::Ordering::Relaxed) as usize;
                                    if m_start >= m as usize { break; }
                                    let bm = std::cmp::min(block_m, m as usize - m_start);
                                    let mut a_band = vec![0.0; bm * bk];
                                    for i in 0..bm {
                                        let global_row_a = m_start + i;
                                        let a_row_start = global_row_a * k as usize + k_start;
                                        a_band[i * bk .. (i + 1) * bk].copy_from_slice(&a_slice[a_row_start .. a_row_start + bk]);
                                    }
                                    unsafe { sgemm(bm, bk, n as usize, 1.0, a_band.as_ptr(), bk as isize, 1, b_band.as_ptr(), n as isize, 1, if k_start == 0 { 0.0 } else { 1.0 }, (out_ptr as *mut f32).add(m_start * n as usize), n as isize, 1); }
                                }
                            });
                        }
                    });
                }
            }
            out
        } else {
            crate::backend::execute_matmul(a_slice, b_slice, m, k, n, self.device == "hybrid")
        };

        Ok(Tensor {
            shape: vec![m as usize, n as usize],
            device: self.device.clone(),
            name: "MatMulResult".to_string(),
            cpu_data: result_data,
            mmap_data: None,
        })
    }

    fn add_into(&mut self, other: &Tensor, out: &mut Tensor) -> PyResult<()> {
        let (out_slice, _) = out.get_slice_raw_mut();
        self.add_into_raw(other, out_slice)
    }

    fn relu_into(&mut self, out: &mut Tensor) -> PyResult<()> {
        let (out_slice, _) = out.get_slice_raw_mut();
        self.activation_into_raw("relu", out_slice)
    }

    fn sigmoid_into(&mut self, out: &mut Tensor) -> PyResult<()> {
        let (out_slice, _) = out.get_slice_raw_mut();
        self.activation_into_raw("sigmoid", out_slice)
    }

    fn silu_into(&mut self, out: &mut Tensor) -> PyResult<()> {
        let (out_slice, _) = out.get_slice_raw_mut();
        self.activation_into_raw("silu", out_slice)
    }

    fn relu(&self) -> PyResult<Tensor> {
        let mut out = Tensor {
            shape: self.shape.clone(), device: self.device.clone(), name: format!("ReLU({})", self.name),
            cpu_data: vec![0.0; self.shape.iter().product()], mmap_data: None,
        };
        let (out_slice, _) = out.get_slice_raw_mut();
        self.activation_into_raw("relu", out_slice)?;
        Ok(out)
    }

    fn sigmoid(&self) -> PyResult<Tensor> {
        let mut out = Tensor {
            shape: self.shape.clone(), device: self.device.clone(), name: format!("Sigmoid({})", self.name),
            cpu_data: vec![0.0; self.shape.iter().product()], mmap_data: None,
        };
        let (out_slice, _) = out.get_slice_raw_mut();
        self.activation_into_raw("sigmoid", out_slice)?;
        Ok(out)
    }

    fn silu(&self) -> PyResult<Tensor> {
        let mut out = Tensor {
            shape: self.shape.clone(), device: self.device.clone(), name: format!("SiLU({})", self.name),
            cpu_data: vec![0.0; self.shape.iter().product()], mmap_data: None,
        };
        let (out_slice, _) = out.get_slice_raw_mut();
        self.activation_into_raw("silu", out_slice)?;
        Ok(out)
    }
}

// Internal Non-Python helpers
impl Tensor {
    fn get_slice_raw(&self) -> (&[f32], bool) {
        if let Some(mmap) = &self.mmap_data {
            (unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) }, true)
        } else {
            (&self.cpu_data, false)
        }
    }

    fn get_slice_raw_mut(&mut self) -> (&mut [f32], bool) {
        if let Some(mmap) = &self.mmap_data {
            (unsafe { std::slice::from_raw_parts_mut(mmap.as_ptr() as *mut f32, mmap.len() / 4) }, true)
        } else {
            (&mut self.cpu_data, false)
        }
    }

    fn add_into_raw(&self, other: &Tensor, out_slice: &mut [f32]) -> PyResult<()> {
        let (a_slice, _) = self.get_slice_raw();
        let (b_slice, _) = other.get_slice_raw();
        if self.device == "cpu" {
            let tile_size = 1024 * 1024;
            if let Some(m) = &self.mmap_data { #[cfg(target_os = "linux")] unsafe { libc::madvise(m.as_ptr() as *mut libc::c_void, m.len(), libc::MADV_SEQUENTIAL); } }
            if let Some(m) = &other.mmap_data { #[cfg(target_os = "linux")] unsafe { libc::madvise(m.as_ptr() as *mut libc::c_void, m.len(), libc::MADV_SEQUENTIAL); } }
            out_slice.par_chunks_mut(tile_size).enumerate().for_each(|(i, o_chunk)| {
                let start = i * tile_size;
                let end = std::cmp::min(start + tile_size, a_slice.len());
                let a_src = &a_slice[start..end];
                let b_src = &b_slice[start..end];
                o_chunk.iter_mut().zip(a_src.iter()).zip(b_src.iter()).for_each(|((c, &a), &b)| *c = a + b);
            });
        } else {
            crate::backend::execute_add_into(a_slice, b_slice, out_slice, self.device == "hybrid");
        }
        Ok(())
    }

    fn activation_into_raw(&self, op: &str, out_slice: &mut [f32]) -> PyResult<()> {
        let (in_slice, _) = self.get_slice_raw();
        if self.device == "cpu" {
            let tile_size = 1024 * 1024;
            if let Some(m) = &self.mmap_data { #[cfg(target_os = "linux")] unsafe { libc::madvise(m.as_ptr() as *mut libc::c_void, m.len(), libc::MADV_SEQUENTIAL); } }
            out_slice.par_chunks_mut(tile_size).enumerate().for_each(|(i, o_chunk)| {
                let start = i * tile_size;
                let end = std::cmp::min(start + tile_size, in_slice.len());
                let i_src = &in_slice[start..end];
                match op {
                    "relu" => o_chunk.iter_mut().zip(i_src.iter()).for_each(|(o, &vi)| *o = if vi > 0.0 { vi } else { 0.0 }),
                    "sigmoid" => o_chunk.iter_mut().zip(i_src.iter()).for_each(|(o, &vi)| *o = 1.0 / (1.0 + (-vi).exp())),
                    "silu" => o_chunk.iter_mut().zip(i_src.iter()).for_each(|(o, &vi)| *o = vi * (1.0 / (1.0 + (-vi).exp()))),
                    _ => {}
                }
            });
        } else {
            crate::backend::execute_activation_into(in_slice, op, out_slice, self.device == "hybrid");
        }
        Ok(())
    }
}
