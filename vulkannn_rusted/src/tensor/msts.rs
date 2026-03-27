use rayon::prelude::*;
use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods};
use super::{Tensor, DataType, IoEngineType};
include!(concat!(env!("OUT_DIR"), "/msts_constants.rs"));

impl Tensor {
    /// Optimal dispatch based on tensor size and hardware constants.
    pub fn execute_unary_op_ssd(&self, op: &str, param1: f32, param2: f32) -> PyResult<Tensor> {
        let bytes_per_elem = self.dtype.size();
        let total_elements = self.shape.iter().product::<usize>();
        let total_bytes = (total_elements * bytes_per_elem) as u64;

        if total_bytes <= DIRECT_MAX as u64 {
            // Path A: Direct (zero threads/atomics)
            self.unary_op_ssd_direct(op, param1, param2)
        } else if total_bytes <= 33_554_432 { // 32MB threshold
            // Path B: Single-thread (1 worker, small tiles in L2)
            self.execute_op_unified(None, op, param1, param2, TILE_SMALL, RING_SMALL, false)
        } else {
            // Path C: Full (2 workers, Rayon parallel, large tiles)
            self.execute_op_unified(None, op, param1, param2, TILE_LARGE, RING_LARGE, true)
        }
    }

    /// Master Dispatcher for binary operations (A + B).
    /// Automatically selects between Direct, Rayon (RAM), or Crook (SSD) paths.
    pub fn dispatch_binary_op(&self, other: &Tensor, op: &str) -> PyResult<Tensor> {
        self.check_shape(other)?;
        
        let is_any_ssd = self.is_ssd() || other.is_ssd();
        let bytes_per_elem = self.dtype.size();
        let total_elements = self.shape.iter().product::<usize>();
        let total_bytes = (total_elements * bytes_per_elem) as u64;

        if !is_any_ssd {
            // RAM-only path: Directly use the CPU ops Smart Dispatcher
            return match op {
                "add" => crate::cpu::ops::binary::add::add_bf16(self, other),
                "sub" => crate::cpu::ops::binary::sub::sub_bf16(self, other),
                "mul" => crate::cpu::ops::binary::mul::mul_bf16(self, other),
                "div" => crate::cpu::ops::binary::div::div_bf16(self, other),
                _ => Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown op: {}", op))),
            };
        }

        // SSD involved: Use the Unified MSTS Orchestrator
        if total_bytes <= DIRECT_MAX as u64 {
             self.binary_op_ssd_direct(other, op)
        } else if total_bytes <= 33_554_432 {
             self.execute_op_unified(Some(other), op, 0.0, 0.0, TILE_SMALL, RING_SMALL, false)
        } else {
             self.execute_op_unified(Some(other), op, 0.0, 0.0, TILE_LARGE, RING_LARGE, true)
        }
    }

    fn binary_op_ssd_direct(&self, other: &Tensor, op: &str) -> PyResult<Tensor> {
        let res_path = format!("{}_{}_{}_dir.ssd", self.name, other.name, op);
        let res_tensor = Self::new_ssd_raw(&res_path, self.shape.clone(), self.dtype)?;

        let total_bytes = (self.shape.iter().product::<usize>() * self.dtype.size()) as usize;
        let mut buf_a = crate::io_uring_engine::AlignedBuffer::new(total_bytes);
        let mut buf_b = crate::io_uring_engine::AlignedBuffer::new(total_bytes);

        // Direct Load of both operands (if they are SSD)
        if self.is_ssd() {
            let engine = match self.ssd_engine.as_ref().unwrap() {
                IoEngineType::ReadOnly(e) | IoEngineType::ReadWrite(e) => e.clone(),
            };
            engine.read_chunk(0, buf_a.as_mut_slice());
        } else {
            // Copy from RAM storage
            self.load_to_buffer(buf_a.as_mut_slice(), 0)?;
        }

        if other.is_ssd() {
            let engine = match other.ssd_engine.as_ref().unwrap() {
                IoEngineType::ReadOnly(e) | IoEngineType::ReadWrite(e) => e.clone(),
            };
            engine.read_chunk(0, buf_b.as_mut_slice());
        } else {
            other.load_to_buffer(buf_b.as_mut_slice(), 0)?;
        }

        // Perform computation using the Serial Leaf Kernels
        match (self.dtype, op) {
            (DataType::BF16, "add") => {
                let a = bytemuck::cast_slice::<u8, half::bf16>(buf_a.as_slice());
                let b = bytemuck::cast_slice::<u8, half::bf16>(buf_b.as_slice());
                let res = bytemuck::cast_slice_mut::<u8, half::bf16>(buf_a.as_mut_slice());
                crate::cpu::ops::binary::add::bf16_add_avx_serial(a, b, res);
            },
            // ... (Add other ops/dtypes here)
            _ => { return Err(pyo3::exceptions::PyValueError::new_err(format!("Direct SSD Binary Op not implemented for {:?} {}", self.dtype, op))); }
        }

        let engine_out = match res_tensor.ssd_engine.as_ref().unwrap() {
            IoEngineType::ReadWrite(e) => e.clone(),
            _ => unreachable!(),
        };
        engine_out.write_chunk(0, buf_a.as_slice());
        Ok(res_tensor)
    }

    /// The Unified MSTS Orchestrator. Handles both Unary and Binary operations
    /// by streaming data through the Capacitor and TensorPool. 
    /// Follows the MERA-400 PPU/CPU decoupling model.
    fn execute_op_unified(&self, other: Option<&Tensor>, op: &str, param1: f32, param2: f32, tile_size: usize, ring_size: usize, _parallel: bool) -> PyResult<Tensor> {
        let res_name = if let Some(o) = other { format!("{}_{}_{}", self.name, o.name, op) } else { format!("{}_{}", self.name, op) };
        let res_path = format!("{}_msts.ssd", res_name);
        let res_tensor = Self::new_ssd_raw(&res_path, self.shape.clone(), self.dtype)?;
        
        let bytes_per_elem = self.dtype.size();
        let total_elements = self.shape.iter().product::<usize>();
        let total_bytes = (total_elements * bytes_per_elem) as u64;

        let cap = Some(crate::tensor::capacitor::get_capacitor());
        let scheduler = crate::crook_scheduler::CrookScheduler::new_custom(ring_size, tile_size, cap);
        
        // 1. Start Reader for Self (Source A)
        if self.is_ssd() {
            let engine = match self.ssd_engine.as_ref().unwrap() {
                IoEngineType::ReadOnly(e) | IoEngineType::ReadWrite(e) => e.clone(),
            };
            crate::crook_scheduler::CrookScheduler::start_read_worker(scheduler.clone(), engine, total_bytes, 0);
        } else {
            // RAM Source: In a true MSTS, we would "fake" a reader or just mark the bit.
            // For now, let's just use the SSD-SSD path for full verification.
        }

        let mut expected_bits = 1;

        // 2. Start Reader for Other (Source B if binary)
        if let Some(other_t) = other {
            if other_t.is_ssd() {
                let engine = match other_t.ssd_engine.as_ref().unwrap() {
                    IoEngineType::ReadOnly(e) | IoEngineType::ReadWrite(e) => e.clone(),
                };
                crate::crook_scheduler::CrookScheduler::start_read_worker(scheduler.clone(), engine, total_bytes, 1);
                expected_bits = 3; // Binary: 0b11
            }
        }

        // 3. Start Writer
        let engine_out = match res_tensor.ssd_engine.as_ref().unwrap() {
            IoEngineType::ReadWrite(e) => e.clone(),
            _ => unreachable!(),
        };
        crate::crook_scheduler::CrookScheduler::start_write_worker(scheduler.clone(), engine_out, total_bytes);
        
        let mut offset = 0;
        let mut tile_idx = 0;
        while offset < total_bytes {
            let tile = &scheduler.ring[tile_idx];
            
            // BARRIER: Wait for ALL sources to be ready (MERA-400 Handshake)
            while tile.ready_bits.load(std::sync::atomic::Ordering::Acquire) != expected_bits {
                std::hint::spin_loop();
            }

            // Enter COMPUTING state
            while tile.state.compare_exchange(
                crate::crook_scheduler::TILE_READING_FROM_DISK,
                crate::crook_scheduler::TILE_COMPUTING,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed
            ).is_err() {
                std::hint::spin_loop();
            }
            
            let bytes_in_tile = std::cmp::min(tile_size, (total_bytes - offset) as usize);
            
            // Execute the Leaf Kernel on the prepared slots
            match (self.dtype, op) {
                (DataType::BF16, "add") if other.is_some() => {
                    let a_ptr = tile.slot_a.get_ptr(bytes_in_tile);
                    let b_ptr = tile.slot_b.get_ptr(bytes_in_tile);
                    let res_ptr = tile.slot_res.get_ptr(bytes_in_tile);
                    
                    let a = unsafe { std::slice::from_raw_parts(a_ptr as *const half::bf16, bytes_in_tile / 2) };
                    let b = unsafe { std::slice::from_raw_parts(b_ptr as *const half::bf16, bytes_in_tile / 2) };
                    let res = unsafe { std::slice::from_raw_parts_mut(res_ptr as *mut half::bf16, bytes_in_tile / 2) };
                    
                    crate::cpu::ops::binary::add::bf16::add_bf16_avx_serial(a, b, res);
                },
                (DataType::BF16, "sub") if other.is_some() => {
                    let a_ptr = tile.slot_a.get_ptr(bytes_in_tile);
                    let b_ptr = tile.slot_b.get_ptr(bytes_in_tile);
                    let res_ptr = tile.slot_res.get_ptr(bytes_in_tile);
                    
                    let a = unsafe { std::slice::from_raw_parts(a_ptr as *const half::bf16, bytes_in_tile / 2) };
                    let b = unsafe { std::slice::from_raw_parts(b_ptr as *const half::bf16, bytes_in_tile / 2) };
                    let res = unsafe { std::slice::from_raw_parts_mut(res_ptr as *mut half::bf16, bytes_in_tile / 2) };
                    
                    crate::cpu::ops::binary::sub::bf16::sub_bf16_avx_serial(a, b, res);
                },
                (DataType::BF16, "mul") if other.is_some() => {
                    let a_ptr = tile.slot_a.get_ptr(bytes_in_tile);
                    let b_ptr = tile.slot_b.get_ptr(bytes_in_tile);
                    let res_ptr = tile.slot_res.get_ptr(bytes_in_tile);
                    
                    let a = unsafe { std::slice::from_raw_parts(a_ptr as *const half::bf16, bytes_in_tile / 2) };
                    let b = unsafe { std::slice::from_raw_parts(b_ptr as *const half::bf16, bytes_in_tile / 2) };
                    let res = unsafe { std::slice::from_raw_parts_mut(res_ptr as *mut half::bf16, bytes_in_tile / 2) };
                    
                    crate::cpu::ops::binary::mul::bf16::mul_bf16_avx_serial(a, b, res);
                },
                (DataType::BF16, "div") if other.is_some() => {
                    let a_ptr = tile.slot_a.get_ptr(bytes_in_tile);
                    let b_ptr = tile.slot_b.get_ptr(bytes_in_tile);
                    let res_ptr = tile.slot_res.get_ptr(bytes_in_tile);
                    
                    let a = unsafe { std::slice::from_raw_parts(a_ptr as *const half::bf16, bytes_in_tile / 2) };
                    let b = unsafe { std::slice::from_raw_parts(b_ptr as *const half::bf16, bytes_in_tile / 2) };
                    let res = unsafe { std::slice::from_raw_parts_mut(res_ptr as *mut half::bf16, bytes_in_tile / 2) };
                    
                    crate::cpu::ops::binary::div::bf16::div_bf16_avx_serial(a, b, res);
                },
                (DataType::BF16, "relu") if other.is_none() => {
                    let a_ptr = tile.slot_a.get_ptr(bytes_in_tile);
                    let res_ptr = tile.slot_res.get_ptr(bytes_in_tile);
                    let a = unsafe { std::slice::from_raw_parts(a_ptr as *const half::bf16, bytes_in_tile / 2) };
                    let res = unsafe { std::slice::from_raw_parts_mut(res_ptr as *mut half::bf16, bytes_in_tile / 2) };
                    for (i, val) in a.iter().enumerate() {
                        res[i] = if val.to_f32() < 0.0 { half::bf16::ZERO } else { *val };
                    }
                },
                _ => {} 
            }
            
            // Signal Writer
            tile.ready_bits.store(0, std::sync::atomic::Ordering::Release); // Clear bits for reuse
            tile.state.store(crate::crook_scheduler::TILE_READY_FOR_WRITE, std::sync::atomic::Ordering::Release);
            
            offset += bytes_in_tile as u64;
            tile_idx = (tile_idx + 1) % ring_size;
        }
        
        Ok(res_tensor)
    }

    /// Extreme I/O MERA-400 architecture for SSD tensors
    pub fn execute_load_to_f32_vec_msts(&self) -> Vec<f32> {
        let engine = match &self.ssd_engine {
            Some(IoEngineType::ReadOnly(e)) => e.clone(),
            Some(IoEngineType::ReadWrite(e)) => e.clone(),
            None => panic!("Not an SSD tensor"),
        };
        
        let total_elems = self.shape.iter().product::<usize>();
        let mut out = vec![0.0; total_elems];
        let bytes_per_elem = match self.dtype {
            DataType::F32 => 4,
            DataType::Int8 => 1,
            _ => 2,
        };
        let total_bytes = (total_elems * bytes_per_elem) as u64;
        
        let cap = Some(crate::tensor::capacitor::get_capacitor());
        let scheduler = crate::crook_scheduler::CrookScheduler::new_custom(8, 8388608, cap); // 64MB ring (8x8MB)
        let io_handle = crate::crook_scheduler::CrookScheduler::start_read_worker(scheduler.clone(), engine, total_bytes);
        
        let mut offset = 0;
        let ring_size = scheduler.ring.len();
        let mut tile_idx = 0;
        
        while offset < total_bytes {
            let tile = &scheduler.ring[tile_idx];
            
            while tile.state.compare_exchange(
                crate::crook_scheduler::TILE_READY_FOR_COMPUTE,
                crate::crook_scheduler::TILE_COMPUTING,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed
            ).is_err() {
                std::hint::spin_loop();
            }
            
            let bytes_in_tile = std::cmp::min(8388608, (total_bytes - offset) as usize);
            let payload = tile.get_data(bytes_in_tile);
            
            match self.dtype {
                DataType::F32 => {
                    let slice = bytemuck::cast_slice::<u8, f32>(payload);
                    let start_idx = (offset / 4) as usize;
                    out[start_idx..start_idx + slice.len()].copy_from_slice(slice);
                },
                DataType::F16 => {
                    let slice = bytemuck::cast_slice::<u8, half::f16>(payload);
                    let start_idx = (offset / 2) as usize;
                    crate::cpu::convert_f16_to_f32(slice, &mut out[start_idx..start_idx + slice.len()]);
                },
                DataType::BF16 => {
                    let slice = bytemuck::cast_slice::<u8, half::bf16>(payload);
                    let start_idx = (offset / 2) as usize;
                    crate::cpu::convert_bf16_to_f32(slice, &mut out[start_idx..start_idx + slice.len()]);
                },
                DataType::Int8 => {
                    let slice = bytemuck::cast_slice::<u8, i8>(payload);
                    let start_idx = offset as usize;
                    for (i, val) in slice.iter().enumerate() { out[start_idx + i] = *val as f32; }
                },
                DataType::BitNet2 | DataType::BitNet1_6 | DataType::I2_S => {
                    // Dequantizing BitNet to F32 for CPU feedback
                    let slice = payload; 
                    let start_idx = offset as usize;
                    // Slow fallback for de-serialization if needed, or just leave as is
                    for (i, &val) in slice.iter().enumerate() { out[start_idx + i] = val as f32; }
                }
            }
            
            tile.state.store(crate::crook_scheduler::TILE_EMPTY, std::sync::atomic::Ordering::Release);
            offset += bytes_in_tile as u64;
            tile_idx = (tile_idx + 1) % ring_size;
        }
        
        io_handle.join().unwrap();
        out
    }



    /// Extreme I/O MSTS PyTorch Fallback.
    /// Executes any PyTorch function/module on 1MB tiles of an SSD tensor.
    /// This allows OxTorch to "borrow" PyTorch's full API for massive tensors.
    pub fn unary_op_msts_pytorch(&self, py: Python, callback: PyObject) -> PyResult<Tensor> {
        let res_path = format!("{}_pt.ssd", self.name);
        let res_tensor = Self::new_ssd_raw(&res_path, self.shape.clone(), self.dtype)?;

        let engine_in = match self.ssd_engine.as_ref().unwrap() {
            IoEngineType::ReadOnly(e) => e.clone(),
            IoEngineType::ReadWrite(e) => e.clone(),
        };
        let engine_out = match res_tensor.ssd_engine.as_ref().unwrap() {
            IoEngineType::ReadWrite(e) => e.clone(),
            _ => unreachable!(),
        };

        let bytes_per_elem = self.dtype.size();
        let total_elements = self.shape.iter().product::<usize>();
        let total_bytes = (total_elements * bytes_per_elem) as u64;

        let ring_size = 8;
        let cap = Some(crate::tensor::capacitor::get_capacitor());
        let scheduler = crate::crook_scheduler::CrookScheduler::new_custom(ring_size, 8388608, cap);
        
        let r_sched = scheduler.clone();
        let w_sched = scheduler.clone();
        let r_handle = crate::crook_scheduler::CrookScheduler::start_read_worker(r_sched, engine_in, total_bytes);
        let w_handle = crate::crook_scheduler::CrookScheduler::start_write_worker(w_sched, engine_out, total_bytes);

        let mut offset = 0;
        let mut tile_idx = 0;
        
        while offset < total_bytes {
            let tile = &scheduler.ring[tile_idx];
            
            // Wait for I/O thread to finish reading
            while tile.state.compare_exchange(
                crate::crook_scheduler::TILE_READY_FOR_COMPUTE,
                crate::crook_scheduler::TILE_COMPUTING,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed
            ).is_err() {
                std::hint::spin_loop();
            }

            let bytes_in_tile = std::cmp::min(8388608, (total_bytes - offset) as usize);
            let slice = tile.get_data_mut(bytes_in_tile);

            // Create NumPy array from slice (safe copy)
            let np_array = match self.dtype {
                DataType::F32 => {
                    let s = bytemuck::cast_slice_mut::<u8, f32>(slice);
                    PyArray1::from_slice_bound(py, s).into_any()
                },
                DataType::Int8 | DataType::BitNet2 | DataType::BitNet1_6 | DataType::I2_S => {
                    let s = bytemuck::cast_slice_mut::<u8, i8>(slice);
                    PyArray1::from_slice_bound(py, s).into_any()
                },
                _ => {
                    // Fallback to u16 for F16/BF16 (Python/Torch will view it accordingly)
                    let s = bytemuck::cast_slice_mut::<u8, u16>(slice);
                    PyArray1::from_slice_bound(py, s).into_any()
                }
            };

            // Call Python callback with the NumPy view. It MUST return the new NumPy array.
            let res_obj = callback.call1(py, (np_array,))?;

            // Extract the returned array and copy its data back into our tile slice
            match self.dtype {
                DataType::F32 => {
                    let res_array: Bound<'_, PyArray1<f32>> = res_obj.extract(py)?;
                    let res_vec = res_array.to_vec()?;
                    let s = bytemuck::cast_slice_mut::<u8, f32>(slice);
                    s.copy_from_slice(&res_vec);
                },
                DataType::Int8 | DataType::BitNet2 | DataType::BitNet1_6 | DataType::I2_S => {
                    let res_array: Bound<'_, PyArray1<i8>> = res_obj.extract(py)?;
                    let res_vec = res_array.to_vec()?;
                    let s = bytemuck::cast_slice_mut::<u8, i8>(slice);
                    s.copy_from_slice(&res_vec);
                },
                DataType::F16 | DataType::BF16 => {
                    let res_array: Bound<'_, PyArray1<u16>> = res_obj.extract(py)?;
                    let res_vec = res_array.to_vec()?;
                    let s = bytemuck::cast_slice_mut::<u8, u16>(slice);
                    s.copy_from_slice(&res_vec);
                }
            }

            // Mark for write worker
            tile.state.store(crate::crook_scheduler::TILE_READY_FOR_WRITE, std::sync::atomic::Ordering::Release);
            
            offset += bytes_in_tile as u64;
            tile_idx = (tile_idx + 1) % ring_size;
        }

        r_handle.join().unwrap();
        w_handle.join().unwrap();
        
        Ok(res_tensor)
    }

    /// True MSTS Streaming Copy.
    /// Streams tiles from this SSD tensor directly into a caller-supplied byte buffer
    /// at `dest_offset_bytes`. The caller allocates RAM only for the final result.
    /// Supports all dtypes (raw byte copy of the correct width).
    pub fn load_to_buffer(&self, dest: &mut [u8], dest_offset_bytes: u64) -> PyResult<()> {
        let engine = match &self.ssd_engine {
            Some(IoEngineType::ReadOnly(e)) => e.clone(),
            Some(IoEngineType::ReadWrite(e)) => e.clone(),
            None => return Err(pyo3::exceptions::PyValueError::new_err("load_to_buffer: not an SSD tensor")),
        };

        let total_elems = self.shape.iter().product::<usize>();
        let bytes_per_elem = self.dtype.size();
        let total_bytes = (total_elems * bytes_per_elem) as u64;

        let ring_size = 8;
        let cap = Some(crate::tensor::capacitor::get_capacitor());
        let scheduler = crate::crook_scheduler::CrookScheduler::new_custom(ring_size, 8388608, cap);
        let io_handle = crate::crook_scheduler::CrookScheduler::start_read_worker(
            scheduler.clone(), engine, total_bytes,
        );

        let mut src_offset: u64 = 0;
        let mut tile_idx: usize = 0;

        while src_offset < total_bytes {
            let tile = &scheduler.ring[tile_idx];

            // Spin until I/O worker signals the tile is ready
            while tile.state.compare_exchange(
                crate::crook_scheduler::TILE_READY_FOR_COMPUTE,
                crate::crook_scheduler::TILE_COMPUTING,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed,
            ).is_err() {
                std::hint::spin_loop();
            }

            let bytes_in_tile = std::cmp::min(1_048_576, (total_bytes - src_offset) as usize);
            let src_slice = tile.get_data(bytes_in_tile);

            // Direct copy into destination window — no intermediate Vec
            let dst_start = (dest_offset_bytes + src_offset) as usize;
            dest[dst_start..dst_start + bytes_in_tile].copy_from_slice(src_slice);

            tile.state.store(crate::crook_scheduler::TILE_EMPTY, std::sync::atomic::Ordering::Release);
            src_offset += bytes_in_tile as u64;
            tile_idx = (tile_idx + 1) % ring_size;
        }

        io_handle.join().unwrap();
        Ok(())
    }

    /// Load the entire SSD tensor into typed CPU RAM storage.
    /// Kept for single-tensor convenience (e.g. to_numpy).
    pub fn execute_load_to_storage_cpu(&self) -> PyResult<crate::tensor::Storage> {
        let total_elems = self.shape.iter().product::<usize>();
        let bytes_per_elem = self.dtype.size();
        let total_bytes = (total_elems * bytes_per_elem) as usize;

        let mut raw = unsafe {
            let cap = (total_bytes + 7) / 8;
            let v_f64 = vec![0.0f64; cap];
            let ptr = v_f64.as_ptr();
            std::mem::forget(v_f64);
            Vec::from_raw_parts(ptr as *mut u8, total_bytes, cap * 8)
        };
        self.load_to_buffer(&mut raw, 0)?;

        Ok(match self.dtype {
            DataType::F32 => unsafe {
                let ptr = raw.as_ptr();
                let len = raw.len() / 4;
                let cap = raw.capacity() / 4;
                std::mem::forget(raw);
                crate::tensor::Storage::F32(Vec::from_raw_parts(ptr as *mut f32, len, cap))
            },
            DataType::F16 => unsafe {
                let ptr = raw.as_ptr();
                let len = raw.len() / 2;
                let cap = raw.capacity() / 2;
                std::mem::forget(raw);
                crate::tensor::Storage::F16(Vec::from_raw_parts(ptr as *mut half::f16, len, cap))
            },
            DataType::BF16 => unsafe {
                let ptr = raw.as_ptr();
                let len = raw.len() / 2;
                let cap = raw.capacity() / 2;
                std::mem::forget(raw);
                crate::tensor::Storage::BF16(Vec::from_raw_parts(ptr as *mut half::bf16, len, cap))
            },
            DataType::Int8 | DataType::BitNet2 | DataType::BitNet1_6 | DataType::I2_S => unsafe {
                let ptr = raw.as_ptr();
                let len = raw.len();
                let cap = raw.capacity();
                std::mem::forget(raw);
                crate::tensor::Storage::Int8(Vec::from_raw_parts(ptr as *mut i8, len, cap))
            },
        })
    }
    /// Proactively pulls SSD data into the Global RAM Capacitor in a background thread.
    /// This eliminates I/O wait times for subsequent MSTS operations.
    pub fn prefetch_ssd(&self) {
        let engine = match &self.ssd_engine {
            Some(IoEngineType::ReadOnly(e)) => e.clone(),
            Some(IoEngineType::ReadWrite(e)) => e.clone(),
            None => return,
        };
        
        let bytes_per_elem = self.dtype.size();
        let total_elements = self.shape.iter().product::<usize>();
        let total_bytes = (total_elements * bytes_per_elem) as u64;
        let capacitor = crate::tensor::capacitor::get_capacitor();
        
        std::thread::spawn(move || {
            let mut offset = 0;
            let mut chunk_id = 0;
            let chunk_size = 8388608; // 8MB chunks
            let max_in_flight = 64;   // Keep up to 512MB in flight (Aggressive for ZFS Stripe)
            let mut pending = std::collections::HashMap::new();
            
            while offset < total_bytes || !pending.is_empty() {
                // 1. Submit new requests if we have space in the flight window
                while offset < total_bytes && pending.len() < max_in_flight {
                    let size = std::cmp::min(chunk_size as u64, total_bytes - offset) as usize;
                    let cap_offset = engine.submit_read_to_capacitor(offset, size, &capacitor, chunk_id);
                    pending.insert(chunk_id, (cap_offset, size));
                    
                    offset += size as u64;
                    chunk_id += 1;
                }
                
                // 2. Poll for completions
                engine.poll_completions(&capacitor, &mut pending);
                
                // 3. Yield to keep CPU free for workers, but check often
                if !pending.is_empty() {
                    std::thread::yield_now();
                }
            }
            println!("[VNN] SSD Parallel Prefetching complete for {} chunks.", chunk_id);
        });
    }
}
