mod types;
mod storage;
mod constructors;
mod access;
mod ops;
mod reductions;
mod linalg;
mod msts;

pub use types::{DataType, IoEngineType};
pub use storage::Storage;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

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
    #[pyo3(signature = (shape, dtype=DataType::F32, device="cpu", name="tensor"))]
    pub fn py_new(shape: Vec<usize>, dtype: DataType, device: &str, name: &str) -> PyResult<Self> {
        Self::new(shape, dtype, device, name)
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
    #[pyo3(signature = (input, weight, bias=None, activation="none"))]
    pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, activation: &str) -> PyResult<Tensor> {
        Self::execute_linear(input, weight, bias, activation)
    }

    #[pyo3(name = "__matmul__")]
    pub fn py_matmul(&self, other: &Tensor) -> PyResult<Tensor> {
         self.__matmul__(other)
    }

    pub fn transpose(&self) -> PyResult<Tensor> {
        self.execute_transpose()
    }

    #[pyo3(name = "__add__")]
    pub fn py_add(&self, other: &Tensor) -> PyResult<Tensor> { self.elementwise_op(other, "add") }
    #[pyo3(name = "__sub__")]
    pub fn py_sub(&self, other: &Tensor) -> PyResult<Tensor> { self.elementwise_op(other, "sub") }
    #[pyo3(name = "__mul__")]
    pub fn py_mul(&self, other: &Tensor) -> PyResult<Tensor> { self.elementwise_op(other, "mul") }
    #[pyo3(name = "__truediv__")]
    pub fn py_div(&self, other: &Tensor) -> PyResult<Tensor> { self.elementwise_op(other, "div") }

    pub fn relu(&self) -> PyResult<Tensor> { self.unary_op("relu", 0.0, 0.0) }
    pub fn sigmoid(&self) -> PyResult<Tensor> { self.unary_op("sigmoid", 0.0, 0.0) }
    pub fn silu(&self) -> PyResult<Tensor> { self.unary_op("silu", 0.0, 0.0) }
    pub fn gelu(&self) -> PyResult<Tensor> { self.unary_op("gelu", 0.0, 0.0) }
    pub fn tanh(&self) -> PyResult<Tensor> { self.unary_op("tanh", 0.0, 0.0) }

    pub fn apply_softmax(&self, dim: i64, is_log: bool) -> PyResult<Tensor> {
        self.execute_softmax(dim, is_log)
    }

    pub fn reduce(&self, op: &str, dim: Option<i64>) -> PyResult<Tensor> {
        self.execute_reduce(op, dim)
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Tensor> {
        self.execute_reshape(new_shape)
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
