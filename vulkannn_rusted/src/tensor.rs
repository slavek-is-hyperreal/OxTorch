use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use numpy::ndarray::Array;

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
    /// Initialize a new Tensor. Currently only supports 1D/2D flat fallbacks for the MVP.
    #[new]
    #[pyo3(signature = (data=None, shape=None, dtype=None, device="auto", name="Tensor"))]
    fn new(
        data: Option<PyReadonlyArrayDyn<'_, f32>>,
        shape: Option<Vec<usize>>,
        dtype: Option<Bound<'_, PyAny>>, // Renamed parameter to match signature
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
        
        Ok(Tensor {
            shape,
            device: "ssd".to_string(),
            name: "SSDMapped".to_string(),
            cpu_data: Vec::new(),
            mmap_data: Some(mmap),
        })
    }

    /// Converts the internal Rust CPU vector into a NumPy array and hands memory to Python.
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        let cpu_slice = if let Some(mmap) = &self.mmap_data {
            let ptr = mmap.as_ptr() as *const f32;
            let len = mmap.len() / 4;
            unsafe { std::slice::from_raw_parts(ptr, len) }
        } else {
            &self.cpu_data
        };

        let array = Array::from_shape_vec(self.shape.clone(), cpu_slice.to_vec())
            .map_err(|e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string()))?;
        Ok(array.into_pyarray_bound(py))
    }

    /// Alias for backwards compatibility with `vulkan_nn_lib` tests.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        self.numpy(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, device='{}', name='{}')",
            self.shape, self.device, self.name
        )
    }

    fn __add__(&self, other: &Tensor) -> PyResult<Tensor> {
        if self.shape != other.shape {
            return Err(PyValueError::new_err("Shapes must match for basic addition MVP"));
        }

        let a_slice = if let Some(mmap) = &self.mmap_data {
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) }
        } else {
            &self.cpu_data
        };

        let b_slice = if let Some(mmap) = &other.mmap_data {
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) }
        } else {
            &other.cpu_data
        };
        
        // Respect VRAM budget before sending to GPU 
        if let Some(mut budgets) = crate::streaming::BUDGETS.get().unwrap().lock().ok() {
            let bytes_needed = a_slice.len() * 4 * 3 ; // A + B + C
            if bytes_needed > budgets.l1_vram_max_bytes {
                println!("WARN: Exceeded Safe L1 VRAM Budget ({} mb needed > {} budget). Streaming required.", bytes_needed / 1024 / 1024, budgets.l1_vram_max_bytes / 1024 / 1024);
            }
        }

        let result_data = crate::backend::execute_add(a_slice, b_slice);
        
        Ok(Tensor {
            shape: self.shape.clone(),
            device: self.device.clone(),
            name: "AddResult".to_string(),
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

        if k != k2 {
            return Err(PyValueError::new_err(format!("Shape mismatch: {:?} x {:?}", self.shape, other.shape)));
        }

        let a_slice = if let Some(mmap) = &self.mmap_data {
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) }
        } else {
            &self.cpu_data
        };

        let b_slice = if let Some(mmap) = &other.mmap_data {
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) }
        } else {
            &other.cpu_data
        };

        let result_data = crate::backend::execute_matmul(a_slice, b_slice, m, k, n);

        Ok(Tensor {
            shape: vec![m as usize, n as usize],
            device: self.device.clone(),
            name: "MatMulResult".to_string(),
            cpu_data: result_data,
            mmap_data: None,
        })
    }

    // -----------------------------------------------------
    // ACTIVATIONS
    // -----------------------------------------------------

    fn relu(&self) -> PyResult<Tensor> {
        let slice = if let Some(mmap) = &self.mmap_data {
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) }
        } else {
            &self.cpu_data
        };
        Ok(Tensor {
            shape: self.shape.clone(),
            device: self.device.clone(),
            name: "ReLU".to_string(),
            cpu_data: crate::backend::execute_activation(slice, "relu"),
            mmap_data: None,
        })
    }

    fn sigmoid(&self) -> PyResult<Tensor> {
        let slice = if let Some(mmap) = &self.mmap_data {
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) }
        } else {
            &self.cpu_data
        };
        Ok(Tensor {
            shape: self.shape.clone(),
            device: self.device.clone(),
            name: "Sigmoid".to_string(),
            cpu_data: crate::backend::execute_activation(slice, "sigmoid"),
            mmap_data: None,
        })
    }

    fn silu(&self) -> PyResult<Tensor> {
        let slice = if let Some(mmap) = &self.mmap_data {
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) }
        } else {
            &self.cpu_data
        };
        Ok(Tensor {
            shape: self.shape.clone(),
            device: self.device.clone(),
            name: "SiLU".to_string(),
            cpu_data: crate::backend::execute_activation(slice, "silu"),
            mmap_data: None,
        })
    }
}
