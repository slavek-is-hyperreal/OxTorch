use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use numpy::ndarray::Array;
use rayon::prelude::*;
use matrixmultiply::sgemm;

pub enum MmapType {
    ReadOnly(std::sync::Arc<memmap2::Mmap>),
    ReadWrite(std::sync::Arc<memmap2::MmapMut>),
}

#[pyclass(unsendable)]
pub struct Tensor {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get, set)]
    pub device: String,
    #[pyo3(get, set)]
    pub name: String,
    pub cpu_data: Vec<f32>,
    pub mmap_data: Option<MmapType>,
}

#[pymethods]
impl Tensor {
    #[new]
    #[pyo3(signature = (data=None, shape=None, dtype=None, device="auto", name="Tensor"))]
    #[allow(unused_variables)]
    fn new(data: Option<PyReadonlyArrayDyn<'_, f32>>, shape: Option<Vec<usize>>, dtype: Option<Bound<'_, PyAny>>, device: &str, name: &str) -> PyResult<Self> {
        let (cpu_data, final_shape) = if let Some(arr) = data {
            let nd_arr = arr.as_array();
            (nd_arr.as_slice().unwrap_or(&[]).to_vec(), nd_arr.shape().to_vec())
        } else if let Some(s) = shape {
            (vec![0.0; s.iter().product()], s)
        } else {
            return Err(PyValueError::new_err("Must provide either data or shape"));
        };
        Ok(Tensor { shape: final_shape, device: device.to_string(), name: name.to_string(), cpu_data, mmap_data: None })
    }

    #[staticmethod]
    fn from_ssd(path: &str, shape: Vec<usize>) -> PyResult<Self> {
        let file = std::fs::File::open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let metadata = file.metadata().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let expected_size = shape.iter().product::<usize>() * 4;
        if metadata.len() < expected_size as u64 {
            return Err(PyValueError::new_err(format!("File size mismatch: expected at least {} bytes, found {}", expected_size, metadata.len())));
        }
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).map_err(|e| PyValueError::new_err(e.to_string()))? };
        #[cfg(target_os = "linux")] unsafe { libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_SEQUENTIAL); }
        Ok(Tensor { shape, device: "ssd".to_string(), name: "SSDMapped".to_string(), cpu_data: Vec::new(), mmap_data: Some(MmapType::ReadOnly(std::sync::Arc::new(mmap))) })
    }

    #[staticmethod]
    fn new_ssd(path: &str, shape: Vec<usize>) -> PyResult<Self> {
        let size = shape.iter().product::<usize>();
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        file.set_len((size * 4) as u64).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file).map_err(|e| PyValueError::new_err(e.to_string()))? };
        #[cfg(target_os = "linux")] unsafe { libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_SEQUENTIAL); }
        Ok(Tensor { shape, device: "ssd".to_string(), name: "SSDResult".to_string(), cpu_data: Vec::new(), mmap_data: Some(MmapType::ReadWrite(std::sync::Arc::new(mmap))) })
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        let (slice, _) = self.get_slice_raw();
        Ok(Array::from_shape_vec(self.shape.clone(), slice.to_vec()).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray_bound(py))
    }

    fn __repr__(&self) -> String { format!("Tensor(shape={:?}, device='{}', ssd={})", self.shape, self.device, self.mmap_data.is_some()) }

    // --- ELEMENT-WISE ---
    fn __add__(&self, other: &Tensor) -> PyResult<Tensor> {
        if self.shape != other.shape { return Err(PyValueError::new_err("Shape mismatch")); }
        let mut out = Tensor {
            shape: self.shape.clone(), device: self.device.clone(), name: format!("({}+{})", self.name, other.name),
            cpu_data: vec![0.0; self.shape.iter().product()], mmap_data: None,
        };
        self.add_into_raw(other, &mut out.cpu_data)?;
        Ok(out)
    }

    fn add_into(&mut self, other: &Tensor, out: &mut Tensor) -> PyResult<()> {
        let (out_slice, _) = out.get_slice_raw_mut();
        self.add_into_raw(other, out_slice)
    }

    fn relu(&self) -> PyResult<Tensor> { self.unary_op("relu") }
    fn relu_into(&mut self, out: &mut Tensor) -> PyResult<()> {
        let (out_slice, _) = out.get_slice_raw_mut();
        self.act_into_raw("relu", out_slice)
    }

    fn sigmoid(&self) -> PyResult<Tensor> { self.unary_op("sigmoid") }
    fn silu(&self) -> PyResult<Tensor> { self.unary_op("silu") }

    // --- MATMUL ---
    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 { return Err(PyValueError::new_err("2D required")); }
        let m = self.shape[0]; let k = self.shape[1]; let k2 = other.shape[0]; let n = other.shape[1];
        if k != k2 { return Err(PyValueError::new_err("K mismatch")); }

        let (a, a_ssd) = self.get_slice_raw();
        let (b, b_ssd) = other.get_slice_raw();

        let res = if self.device == "cpu" {
            let mut out = vec![0.0; m * n];
            crate::streaming::init_budgets();
            let safe_ram = crate::streaming::BUDGETS.get().unwrap().lock().unwrap().l2_ram_max_bytes;
            let sz = (m*k + k*n + m*n) * 4;

            if sz < safe_ram {
                if a_ssd || b_ssd {
                    let ar = if a_ssd { self.par_copy(a) } else { a.to_vec() };
                    let br = if b_ssd { self.par_copy(b) } else { b.to_vec() };
                    self.cpu_sgemm(&ar, &br, &mut out, m, k, n);
                } else {
                    self.cpu_sgemm(a, b, &mut out, m, k, n);
                }
            } else {
                self.cpu_sgemm_streamed(a, b, &mut out, m, k, n);
            }
            out
        } else {
            crate::backend::execute_matmul(a, b, m as u32, k as u32, n as u32, self.device == "hybrid")
        };
        Ok(Tensor { shape: vec![m, n], device: self.device.clone(), name: "MatMulRes".to_string(), cpu_data: res, mmap_data: None })
    }
}

// Internal Logic
impl Tensor {
    fn unary_op(&self, op: &str) -> PyResult<Tensor> {
        let mut out = Tensor {
            shape: self.shape.clone(), device: self.device.clone(), name: format!("{}({})", op, self.name),
            cpu_data: vec![0.0; self.shape.iter().product()], mmap_data: None,
        };
        self.act_into_raw(op, &mut out.cpu_data)?;
        Ok(out)
    }

    fn get_slice_raw(&self) -> (&[f32], bool) {
        if let Some(mmap) = &self.mmap_data {
            match mmap {
                MmapType::ReadOnly(m) => (unsafe { std::slice::from_raw_parts(m.as_ptr() as *const f32, m.len()/4) }, true),
                MmapType::ReadWrite(m) => (unsafe { std::slice::from_raw_parts(m.as_ptr() as *const f32, m.len()/4) }, true),
            }
        } else { (&self.cpu_data, false) }
    }

    fn get_slice_raw_mut(&mut self) -> (&mut [f32], bool) {
        if let Some(mmap) = &self.mmap_data {
            match mmap {
                MmapType::ReadWrite(m) => (unsafe { std::slice::from_raw_parts_mut(m.as_ptr() as *mut f32, m.len()/4) }, true),
                MmapType::ReadOnly(_) => panic!("Modify ReadOnly SSD!"),
            }
        } else { (&mut self.cpu_data, false) }
    }

    fn add_into_raw(&self, other: &Tensor, out: &mut [f32]) -> PyResult<()> {
        let (a, a_ssd) = self.get_slice_raw();
        let (b, b_ssd) = other.get_slice_raw();
        if self.device == "cpu" {
            // FAST PATH: Pure RAM
            if !a_ssd && !b_ssd {
                out.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, &av), &bv)| *c = av + bv);
            } else {
                // SSD PATH: Use larger chunks to consolidate IO
                let tile = 1024 * 1024 / 4; 
                out.par_chunks_mut(tile).enumerate().for_each(|(i, chunk)| {
                    let start = i * tile;
                    let asub = &a[start..start + chunk.len()];
                    let bsub = &b[start..start + chunk.len()];
                    for k in 0..chunk.len() { chunk[k] = asub[k] + bsub[k]; }
                });
            }
        } else {
            crate::backend::execute_add_into(a, b, out, self.device == "hybrid", a_ssd || b_ssd);
        }
        Ok(())
    }

    fn act_into_raw(&self, op: &str, out: &mut [f32]) -> PyResult<()> {
        let (i_s, i_ssd) = self.get_slice_raw();
        if self.device == "cpu" {
            // FAST PATH: Pure RAM
            if !i_ssd {
                match op {
                    "relu" => out.par_iter_mut().zip(i_s.par_iter()).for_each(|(o, &i)| *o = if i > 0.0 { i } else { 0.0 }),
                    "sigmoid" => out.par_iter_mut().zip(i_s.par_iter()).for_each(|(o, &i)| *o = 1.0 / (1.0 + (-i).exp())),
                    "silu" => out.par_iter_mut().zip(i_s.par_iter()).for_each(|(o, &i)| *o = i * (1.0 / (1.0 + (-i).exp()))),
                    _ => {}
                }
            } else {
                // SSD PATH
                let tile = 1024 * 1024 / 4;
                out.par_chunks_mut(tile).enumerate().for_each(|(idx, chunk)| {
                    let s = idx * tile;
                    let end = std::cmp::min(s + chunk.len(), i_s.len());
                    if s < end {
                        let isub = &i_s[s..end];
                        // Also make sure chunk and isub have same length to avoid panic later
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
        } else {
            crate::backend::execute_activation_into(i_s, op, out, self.device == "hybrid", i_ssd);
        }
        Ok(())
    }

    fn par_copy(&self, src: &[f32]) -> Vec<f32> {
        let mut dst = vec![0.0; src.len()];
        dst.par_chunks_mut(1024*1024).enumerate().for_each(|(i, c)| {
            let s = i * 1024 * 1024;
            c.copy_from_slice(&src[s..s+c.len()]);
        });
        dst
    }

    fn cpu_sgemm(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
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
                        unsafe { sgemm(cur_m, k, n, 1.0, (ptr_a as *const f32).add(ms * k), k as isize, 1, ptr_b as *const f32, n as isize, 1, 0.0, (ptr_out as *mut f32).add(ms * n), n as isize, 1); }
                    }
                });
            }
        });
    }

    fn cpu_sgemm_streamed(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        let blk_m = 512; let blk_k = 2048;
        for ks in (0..k).step_by(blk_k) {
            let bk = (k - ks).min(blk_k);
            let mut b_band = vec![0.0; bk * n];
            b_band.par_chunks_mut(n).enumerate().for_each(|(i, r)| r.copy_from_slice(&b[(ks + i) * n .. (ks + i + 1) * n]));
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
                            for i in 0..cur_m { a_band[i * bk .. (i + 1) * bk].copy_from_slice(&a[(ms + i) * k + ks .. (ms + i) * k + ks + bk]); }
                            unsafe { sgemm(cur_m, bk, n, 1.0, a_band.as_ptr(), bk as isize, 1, b_band.as_ptr(), n as isize, 1, if ks == 0 { 0.0 } else { 1.0 }, (ptr_out as *mut f32).add(ms * n), n as isize, 1); }
                        }
                    });
                }
            });
        }
    }
}
