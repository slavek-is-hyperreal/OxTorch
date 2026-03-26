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
    BitNet2,   // 2-bit packing: 4 trits per byte (Shannon limit ~79% efficient)
    BitNet1_6, // 1.6-bit packing: 5 trits per byte (Shannon limit ~99.1% efficient)
    #[allow(non_camel_case_types)]
    I2_S,      // GGML type 36 (64 weights per 16 bytes + 4 byte scale = 20 byte block)
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::Int8 => 1,
            DataType::BitNet2 => 0, // Packed type: size depends on shape, handled in storage
            DataType::BitNet1_6 => 0, // Packed type
            DataType::I2_S => 0, // Packed GGML block type
        }
    }
}
