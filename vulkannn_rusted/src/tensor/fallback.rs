use pyo3::prelude::*;
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use super::Tensor;

pub fn call_pytorch_op(op_name: &str, input: &Tensor) -> PyResult<Tensor> {
    Python::with_gil(|py| {
        let torch = py.import_bound("torch")?;
        
        // Convert VNN Tensor to Numpy for PyTorch ingestion
        let np_data = input.to_numpy(py)?;
        let torch_input = torch.call_method1("from_numpy", (np_data,))?;
        
        // Call the operation (e.g., torch.relu, torch.exp, etc.)
        let torch_out = torch.call_method1(op_name, (torch_input,))?;
        
        // Convert back to Numpy, then to VNN Tensor
        let np_out = torch_out.call_method0("numpy")?;
        let array = np_out.extract::<Bound<'_, numpy::PyArrayDyn<f32>>>()?;
        let shape = array.shape().to_vec();
        let vec = array.readonly().to_owned_array().into_raw_vec();
        
        Tensor::new_from_vec(vec, shape, input.dtype, &input.device, &format!("{}_fallback", op_name))
    })
}

// Specialized fallback for MatMul
pub fn call_pytorch_matmul(a: &Tensor, b: &Tensor) -> PyResult<Tensor> {
    Python::with_gil(|py| {
        let torch = py.import_bound("torch")?;
        let a_pt = torch.call_method1("from_numpy", (a.to_numpy(py)?,))?;
        let b_pt = torch.call_method1("from_numpy", (b.to_numpy(py)?,))?;
        
        let out_pt = torch.call_method1("matmul", (a_pt, b_pt))?;
        
        let np_out = out_pt.call_method0("numpy")?;
        let array = np_out.extract::<Bound<'_, numpy::PyArrayDyn<f32>>>()?;
        Tensor::new_from_vec(array.readonly().to_owned_array().into_raw_vec(), array.shape().to_vec(), a.dtype, &a.device, "matmul_fallback")
    })
}
