use pyo3::prelude::*;
use std::sync::Arc;
use crate::io_uring_engine::DirectIoEngine;

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
    Ternary, // BitNet 1.58b: {-1, 0, 1}
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::Int8 | DataType::Ternary => 1,
        }
    }
}
