use std::sync::atomic::{AtomicU32, Ordering, AtomicPtr};
use std::cell::UnsafeCell;
use std::sync::Arc;
use crate::tensor::GiantCapacitor;

pub const TILE_EMPTY: u32 = 0;
pub const TILE_READING_FROM_DISK: u32 = 1;
pub const TILE_READY_FOR_COMPUTE: u32 = 2;
pub const TILE_COMPUTING: u32 = 3;
pub const TILE_READY_FOR_WRITE: u32 = 4;
pub const TILE_WRITING_TO_DISK: u32 = 5;

/// MERA-400 CROOK OS Inspired Stateful Tile
/// Lockless Tagged-Token architecture.
pub struct StatefulTile {
    pub state: AtomicU32,
    pub tile_id: AtomicU32,
    pub payload: UnsafeCell<crate::io_uring_engine::AlignedBuffer>,
    pub capacitor_ptr: AtomicPtr<u8>,
}

unsafe impl Sync for StatefulTile {}
unsafe impl Send for StatefulTile {}

impl StatefulTile {
    /// Constructs a clear, EMPTY Stateful Tile ready for I/O ingestion.
    pub fn new(size: usize) -> Self {
        Self {
            state: AtomicU32::new(TILE_EMPTY),
            tile_id: AtomicU32::new(0),
            payload: UnsafeCell::new(crate::io_uring_engine::AlignedBuffer::new(size)),
            capacitor_ptr: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Returns the data slice, either from the internal AlignedBuffer or Zero-Copy from Capacitor.
    pub fn get_data(&self, size: usize) -> &[u8] {
        let ptr = self.capacitor_ptr.load(Ordering::Acquire);
        if !ptr.is_null() {
            unsafe { std::slice::from_raw_parts(ptr, size) }
        } else {
            &unsafe { &*self.payload.get() }.as_slice()[..size]
        }
    }

    /// Returns a mutable data slice.
    pub fn get_data_mut(&self, size: usize) -> &mut [u8] {
        let ptr = self.capacitor_ptr.load(Ordering::Acquire);
        if !ptr.is_null() {
            unsafe { std::slice::from_raw_parts_mut(ptr, size) }
        } else {
            &mut unsafe { &mut *self.payload.get() }.as_mut_slice()[..size]
        }
    }
}

/// Manages a continuous flow of `StatefulTile` buffers, executing lockless I/O
/// reads and writes decoupled from primary execution threads.
pub struct CrookScheduler {
    pub ring: Vec<Box<StatefulTile>>,
    pub tile_size: usize,
    pub capacitor: Option<Arc<GiantCapacitor>>,
}

impl CrookScheduler {
    /// Instantiates a new scheduler holding an allocated continuous circular cache of given size.
    pub fn new(ring_size: usize) -> Arc<Self> {
        Self::new_custom(ring_size, 8388608, None) // Default 8MB
    }

    pub fn new_custom(ring_size: usize, tile_size: usize, capacitor: Option<Arc<GiantCapacitor>>) -> Arc<Self> {
        let mut ring = Vec::with_capacity(ring_size);
        for _ in 0..ring_size {
            ring.push(Box::new(StatefulTile::new(tile_size)));
        }
        Arc::new(Self { ring, tile_size, capacitor })
    }
    
    /// Starts the autonomous reading worker (Peripheral Processor - PPU).
    pub fn start_read_worker(scheduler: std::sync::Arc<Self>, engine: std::sync::Arc<crate::io_uring_engine::DirectIoEngine>, total_bytes: u64) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut offset = 0;
            let mut tile_idx = 0;
            let ring_size = scheduler.ring.len();
            let mut current_id = 0;
            
            while offset < total_bytes {
                let tile = &scheduler.ring[tile_idx];

                // Spin until tile is EMPTY
                while tile.state.compare_exchange(
                    TILE_EMPTY, 
                    TILE_READING_FROM_DISK, 
                    Ordering::Acquire, 
                    Ordering::Relaxed
                ).is_err() {
                    std::hint::spin_loop();
                }
                
                let bytes_to_read = std::cmp::min(scheduler.tile_size as u64, total_bytes - offset);

                // Option A: Zero-Copy via Capacitor
                if let Some(cap) = &scheduler.capacitor {
                   if let Some(ptr) = cap.get_chunk_ptr(current_id as usize) {
                       tile.capacitor_ptr.store(ptr as *mut u8, Ordering::Release);
                       tile.tile_id.store(current_id, Ordering::Relaxed);
                       tile.state.store(TILE_READY_FOR_COMPUTE, Ordering::Release);
                       offset += bytes_to_read;
                       current_id += 1;
                       tile_idx = (tile_idx + 1) % ring_size;
                       continue;
                   }
                }

                // Option B: Standard Read-through
                tile.capacitor_ptr.store(std::ptr::null_mut(), Ordering::Release);
                let payload_slice = unsafe { 
                    let buf = &mut *tile.payload.get();
                    &mut buf.as_mut_slice()[0..bytes_to_read as usize]
                };
                
                engine.read_chunk(offset, payload_slice);
                
                tile.tile_id.store(current_id, Ordering::Relaxed);
                tile.state.store(TILE_READY_FOR_COMPUTE, Ordering::Release);
                
                offset += bytes_to_read;
                current_id += 1;
                tile_idx = (tile_idx + 1) % ring_size;
            }
        })
    }

    /// Starts the autonomous writing worker.
    pub fn start_write_worker(scheduler: std::sync::Arc<Self>, engine: std::sync::Arc<crate::io_uring_engine::DirectIoEngine>, total_bytes: u64) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut offset = 0;
            let mut tile_idx = 0;
            let ring_size = scheduler.ring.len();
            
            while offset < total_bytes {
                let tile = &scheduler.ring[tile_idx];

                // Spin until tile is READY_FOR_WRITE
                while tile.state.compare_exchange(
                    TILE_READY_FOR_WRITE, 
                    TILE_WRITING_TO_DISK, 
                    Ordering::Acquire, 
                    Ordering::Relaxed
                ).is_err() {
                    std::hint::spin_loop();
                }
                
                let bytes_to_write = std::cmp::min(scheduler.tile_size as u64, total_bytes - offset);
                let payload_slice = tile.get_data(bytes_to_write as usize);
                
                engine.write_chunk(offset, payload_slice);
                
                // Mark as EMPTY for the read worker to reuse
                tile.state.store(TILE_EMPTY, Ordering::Release);
                
                offset += bytes_to_write;
                tile_idx = (tile_idx + 1) % ring_size;
            }
        })
    }
}
