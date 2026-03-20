mod types;
mod storage;
mod constructors;
mod access;
mod ops;
mod reductions;
mod linalg;
mod msts;
mod fallback;

pub use types::{DataType, IoEngineType};
pub use storage::Storage;

use pyo3::prelude::*;
use numpy::{PyArrayMethods, PyUntypedArrayMethods};

#[pyclass]
#[derive(Clone)]
pub struct Tensor {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get, set)]
    pub device: String,
    #[pyo3(get, set)]
    pub name: String,
    pub is_transposed: bool,
    #[pyo3(get)]
    pub dtype: DataType,
    pub storage: Storage,
    pub mmap_data: Option<IoEngineType>,
}

#[pymethods]
impl Tensor {
    #[new]
    #[pyo3(signature = (shape=None, data=None, dtype=DataType::F32, device="cpu", name="tensor"))]
    pub fn py_new(
        shape: Option<Vec<usize>>,
        data: Option<Bound<'_, numpy::PyArrayDyn<f32>>>,
        dtype: DataType,
        device: &str,
        name: &str,
    ) -> PyResult<Self> {
        if let Some(d) = data {
            let shape = d.shape().to_vec();
            // Use readonly() to get an array view, then convert to vec
            let vec = d.readonly().to_owned_array().into_raw_vec();
            Self::new_from_vec(vec, shape, dtype, device, name)
        } else if let Some(s) = shape {
            Self::new(s, dtype, device, name)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Either shape or data must be provided"))
        }
    }

    #[staticmethod]
    pub fn zeros(shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        Self::new_zeros(shape, dtype, device)
    }

    #[staticmethod]
    pub fn ones(shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        Self::new_ones(shape, dtype, device)
    }

    #[staticmethod]
    pub fn rand(shape: Vec<usize>, dtype: DataType, device: &str) -> PyResult<Self> {
        Self::new_rand(shape, dtype, device)
    }

    #[staticmethod]
    pub fn from_ssd(path: &str, shape: Vec<usize>, dtype: DataType) -> PyResult<Self> {
        Self::new_from_ssd(path, shape, dtype)
    }

    #[staticmethod]
    pub fn new_ssd(path: &str, shape: Vec<usize>, dtype: DataType) -> PyResult<Self> {
        Self::new_ssd_raw(path, shape, dtype)
    }

    #[pyo3(signature = (weight, bias=None, activation="none"))]
    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>, activation: &str) -> PyResult<Tensor> {
        Self::execute_linear(self, weight, bias, activation)
    }

    #[pyo3(signature = (weight, scale, bias=None))]
    pub fn bit_linear(&self, weight: &Tensor, scale: &Tensor, bias: Option<&Tensor>) -> PyResult<Tensor> {
        Self::execute_bit_linear(self, weight, scale, bias)
    }

    #[pyo3(name = "__matmul__")]
    pub fn py_matmul(&self, other: &Tensor) -> PyResult<Tensor> {
         self.__matmul__(other)
    }
    
    #[pyo3(name = "bmm")]
    pub fn py_bmm(&self, other: &Tensor) -> PyResult<Tensor> {
         self.bmm(other)
    }

    #[pyo3(name = "layer_norm", signature = (normalized_shape, weight=None, bias=None, eps=1e-5))]
    pub fn py_layer_norm(&self, normalized_shape: Vec<usize>, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> PyResult<Tensor> {
         self.layer_norm(normalized_shape, weight, bias, eps)
    }

    #[pyo3(name = "rms_norm", signature = (normalized_shape, weight=None, eps=1e-5))]
    pub fn py_rms_norm(&self, normalized_shape: Vec<usize>, weight: Option<&Tensor>, eps: f32) -> PyResult<Tensor> {
         self.rms_norm(normalized_shape, weight, eps)
    }

    pub fn transpose(&self) -> PyResult<Tensor> {
        self.execute_transpose()
    }

    #[pyo3(name = "__add__")]
    pub fn py_add(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(t) = other.extract::<Tensor>() { self.elementwise_op(&t, "add") }
        else if let Ok(s) = other.extract::<f64>() { self.scalar_elementwise_op(s, "add") }
        else { Err(pyo3::exceptions::PyTypeError::new_err("Expected Tensor or float")) }
    }
    #[pyo3(name = "__sub__")]
    pub fn py_sub(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(t) = other.extract::<Tensor>() { self.elementwise_op(&t, "sub") }
        else if let Ok(s) = other.extract::<f64>() { self.scalar_elementwise_op(s, "sub") }
        else { Err(pyo3::exceptions::PyTypeError::new_err("Expected Tensor or float")) }
    }
    #[pyo3(name = "__mul__")]
    pub fn py_mul(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(t) = other.extract::<Tensor>() { self.elementwise_op(&t, "mul") }
        else if let Ok(s) = other.extract::<f64>() { self.scalar_elementwise_op(s, "mul") }
        else { Err(pyo3::exceptions::PyTypeError::new_err("Expected Tensor or float")) }
    }
    #[pyo3(name = "__truediv__")]
    pub fn py_div(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(t) = other.extract::<Tensor>() { self.elementwise_op(&t, "div") }
        else if let Ok(s) = other.extract::<f64>() { self.scalar_elementwise_op(s, "div") }
        else { Err(pyo3::exceptions::PyTypeError::new_err("Expected Tensor or float")) }
    }
    #[pyo3(name = "__rmul__")]
    pub fn py_rmul(&self, other: f64) -> PyResult<Tensor> { self.scalar_elementwise_op(other, "mul") }
    #[pyo3(name = "__radd__")]
    pub fn py_radd(&self, other: f64) -> PyResult<Tensor> { self.scalar_elementwise_op(other, "add") }

    pub fn relu(&self) -> PyResult<Tensor> { self.unary_op("relu", 0.0, 0.0) }
    pub fn relu_into(&self, target: &mut Tensor) -> PyResult<()> { self.unary_op_into(target, "relu", 0.0, 0.0) }
    pub fn sigmoid(&self) -> PyResult<Tensor> { self.unary_op("sigmoid", 0.0, 0.0) }
    pub fn silu(&self) -> PyResult<Tensor> { self.unary_op("silu", 0.0, 0.0) }
    pub fn gelu(&self) -> PyResult<Tensor> { self.unary_op("gelu", 0.0, 0.0) }
    pub fn gelu_into(&self, target: &mut Tensor) -> PyResult<()> { self.unary_op_into(target, "gelu", 0.0, 0.0) }
    pub fn tanh(&self) -> PyResult<Tensor> { self.unary_op("tanh", 0.0, 0.0) }

    pub fn apply_softmax(&self, dim: i64, is_log: bool) -> PyResult<Tensor> {
        self.execute_softmax(dim, is_log)
    }

    pub fn softmax(&self, dim: i64) -> PyResult<Tensor> {
        self.apply_softmax(dim, false)
    }

    #[pyo3(signature = (op, dim=None))]
    pub fn reduce(&self, op: &str, dim: Option<i64>) -> PyResult<Tensor> {
        self.execute_reduce(op, dim)
    }

    #[pyo3(signature = (dim=None))]
    pub fn sum(&self, dim: Option<i64>) -> PyResult<Tensor> {
        self.reduce("sum", dim)
    }

    #[pyo3(signature = (dim=None))]
    pub fn mean(&self, dim: Option<i64>) -> PyResult<Tensor> {
        self.reduce("mean", dim)
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Tensor> {
        self.execute_reshape(new_shape)
    }

    pub fn msts_pytorch_apply(&self, py: Python, callback: PyObject) -> PyResult<Tensor> {
        self.unary_op_msts_pytorch(py, callback)
    }
    
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        self.execute_to_numpy(py)
    }
}

impl Tensor {
    pub fn is_ssd(&self) -> bool {
        self.mmap_data.is_some()
    }

    pub fn check_shape(&self, other: &Tensor) -> PyResult<()> {
        if self.shape != other.shape { 
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Shape mismatch: {:?} vs {:?}", self.shape, other.shape))); 
        }
        if self.dtype != other.dtype { 
            return Err(pyo3::exceptions::PyValueError::new_err(format!("DType mismatch: {:?} vs {:?}", self.dtype, other.dtype))); 
        }
        Ok(())
    }
}
