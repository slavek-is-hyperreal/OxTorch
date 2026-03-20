use rayon::prelude::*;
use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods};
use super::{Tensor, DataType, IoEngineType};

impl Tensor {
    /// Extreme I/O MERA-400 architecture for SSD tensors
    pub fn load_to_f32_vec_msts(&self) -> Vec<f32> {
        let engine = match &self.mmap_data {
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
        
        let scheduler = crate::crook_scheduler::CrookScheduler::new(8); // 8MB ring
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
            
            let bytes_in_tile = std::cmp::min(1048576, (total_bytes - offset) as usize);
            let payload = unsafe { &*tile.payload.get() };
            
            match self.dtype {
                DataType::F32 => {
                    let slice = bytemuck::cast_slice::<u8, f32>(&payload[..bytes_in_tile]);
                    let start_idx = (offset / 4) as usize;
                    out[start_idx..start_idx + slice.len()].copy_from_slice(slice);
                },
                DataType::F16 => {
                    let slice = bytemuck::cast_slice::<u8, half::f16>(&payload[..bytes_in_tile]);
                    let start_idx = (offset / 2) as usize;
                    crate::cpu::convert_f16_to_f32(slice, &mut out[start_idx..start_idx + slice.len()]);
                },
                DataType::BF16 => {
                    let slice = bytemuck::cast_slice::<u8, half::bf16>(&payload[..bytes_in_tile]);
                    let start_idx = (offset / 2) as usize;
                    crate::cpu::convert_bf16_to_f32(slice, &mut out[start_idx..start_idx + slice.len()]);
                },
                DataType::Int8 => {
                    let slice = bytemuck::cast_slice::<u8, i8>(&payload[..bytes_in_tile]);
                    let start_idx = offset as usize;
                    for (i, val) in slice.iter().enumerate() { out[start_idx + i] = *val as f32; }
                },
                DataType::Ternary => {
                    let slice = bytemuck::cast_slice::<u8, i8>(&payload[..bytes_in_tile]);
                    let start_idx = offset as usize;
                    for (i, val) in slice.iter().enumerate() { out[start_idx + i] = *val as f32; }
                }
            }
            
            tile.state.store(crate::crook_scheduler::TILE_EMPTY, std::sync::atomic::Ordering::Release);
            offset += bytes_in_tile as u64;
            tile_idx = (tile_idx + 1) % ring_size;
        }
        
        io_handle.join().unwrap();
        out
    }

    pub fn unary_op_ssd(&self, op: &str, param1: f32, param2: f32) -> PyResult<Tensor> {
        let res_path = format!("{}_{}.ssd", self.name, op);
        let res_tensor = Self::new_ssd_raw(&res_path, self.shape.clone(), self.dtype)?;
        
        let engine_in = match self.mmap_data.as_ref().unwrap() {
            IoEngineType::ReadOnly(e) => e.clone(),
            IoEngineType::ReadWrite(e) => e.clone(),
        };
        let engine_out = match res_tensor.mmap_data.as_ref().unwrap() {
            IoEngineType::ReadWrite(e) => e.clone(),
            _ => unreachable!(),
        };
        
        let bytes_per_elem = match self.dtype {
            DataType::F32  => 4,
            DataType::Int8 => 1,
            _ => 2,
        };
        let total_elements = self.shape.iter().product::<usize>();
        let total_bytes = (total_elements * bytes_per_elem) as u64;
        
        let ring_size = 8;
        let scheduler = crate::crook_scheduler::CrookScheduler::new(ring_size);
        let r_sched = scheduler.clone();
        let w_sched = scheduler.clone();
        let r_handle = crate::crook_scheduler::CrookScheduler::start_read_worker(r_sched, engine_in, total_bytes);
        let w_handle = crate::crook_scheduler::CrookScheduler::start_write_worker(w_sched, engine_out, total_bytes);
        
        let mut offset = 0;
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
            
            let bytes_in_tile = std::cmp::min(1048576, (total_bytes - offset) as usize);
            let payload = unsafe { &mut *tile.payload.get() };
            
            match self.dtype {
                DataType::F32 => {
                    let slice = bytemuck::cast_slice_mut::<u8, f32>(&mut payload[..bytes_in_tile]);
                    Self::act_into_raw_parallel_f32(slice, op, param1, param2);
                },
                DataType::F16 => {
                    let slice = bytemuck::cast_slice_mut::<u8, half::f16>(&mut payload[..bytes_in_tile]);
                    slice.par_iter_mut().for_each(|x| { if op == "relu" && x.to_f32() < 0.0 { *x = half::f16::ZERO; } });
                },
                DataType::BF16 => {
                    let slice = bytemuck::cast_slice_mut::<u8, half::bf16>(&mut payload[..bytes_in_tile]);
                    slice.par_iter_mut().for_each(|x| { if op == "relu" && x.to_f32() < 0.0 { *x = half::bf16::ZERO; } });
                },
                DataType::Int8 => {
                    let slice = bytemuck::cast_slice_mut::<u8, i8>(&mut payload[..bytes_in_tile]);
                    slice.par_iter_mut().for_each(|x| { if op == "relu" && *x < 0 { *x = 0; } });
                },
                DataType::Ternary => {
                    let slice = bytemuck::cast_slice_mut::<u8, i8>(&mut payload[..bytes_in_tile]);
                    slice.par_iter_mut().for_each(|x| { if op == "relu" && *x < 0 { *x = 0; } });
                }
            }
            
            tile.state.store(crate::crook_scheduler::TILE_READY_FOR_WRITE, std::sync::atomic::Ordering::Release);
            offset += bytes_in_tile as u64;
            tile_idx = (tile_idx + 1) % ring_size;
        }
        r_handle.join().unwrap();
        w_handle.join().unwrap();
        Ok(res_tensor)
    }

    /// Extreme I/O MSTS PyTorch Fallback.
    /// Executes any PyTorch function/module on 1MB tiles of an SSD tensor.
    /// This allows OxTorch to "borrow" PyTorch's full API for massive tensors.
    pub fn unary_op_msts_pytorch(&self, py: Python, callback: PyObject) -> PyResult<Tensor> {
        let res_path = format!("{}_pt.ssd", self.name);
        let res_tensor = Self::new_ssd_raw(&res_path, self.shape.clone(), self.dtype)?;

        let engine_in = match self.mmap_data.as_ref().unwrap() {
            IoEngineType::ReadOnly(e) => e.clone(),
            IoEngineType::ReadWrite(e) => e.clone(),
        };
        let engine_out = match res_tensor.mmap_data.as_ref().unwrap() {
            IoEngineType::ReadWrite(e) => e.clone(),
            _ => unreachable!(),
        };

        let bytes_per_elem = self.dtype.size();
        let total_elements = self.shape.iter().product::<usize>();
        let total_bytes = (total_elements * bytes_per_elem) as u64;

        let ring_size = 8;
        let scheduler = crate::crook_scheduler::CrookScheduler::new(ring_size);
        
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

            let bytes_in_tile = std::cmp::min(1048576, (total_bytes - offset) as usize);
            let payload = unsafe { &mut *tile.payload.get() };
            let slice = &mut payload[..bytes_in_tile];

            // Create NumPy array from slice (safe copy)
            let np_array = match self.dtype {
                DataType::F32 => {
                    let s = bytemuck::cast_slice_mut::<u8, f32>(slice);
                    PyArray1::from_slice_bound(py, s).into_any()
                },
                DataType::Int8 => {
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
                DataType::Int8 | DataType::Ternary => {
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
}
