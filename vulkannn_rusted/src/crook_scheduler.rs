use std::sync::atomic::{AtomicU32, Ordering};
use std::cell::UnsafeCell;

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
        }
    }
}

/// Manages a continuous flow of `StatefulTile` buffers, executing lockless I/O
/// reads and writes decoupled from primary execution threads.
pub struct CrookScheduler {
    pub ring: Vec<Box<StatefulTile>>,
    pub tile_size: usize,
}

impl CrookScheduler {
    /// Instantiates a new scheduler holding an allocated continuous circular cache of given size.
    pub fn new(ring_size: usize) -> std::sync::Arc<Self> {
        Self::new_custom(ring_size, 1048576) // Default 1MB
    }

    pub fn new_custom(ring_size: usize, tile_size: usize) -> std::sync::Arc<Self> {
        let mut ring = Vec::with_capacity(ring_size);
        for _ in 0..ring_size {
            ring.push(Box::new(StatefulTile::new(tile_size)));
        }
        std::sync::Arc::new(Self { ring, tile_size })
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
                let payload_slice = unsafe { 
                    let buf = &*tile.payload.get();
                    &buf.as_slice()[0..bytes_to_write as usize]
                };
                
                engine.write_chunk(offset, payload_slice);
                
                // Mark as EMPTY for the read worker to reuse
                tile.state.store(TILE_EMPTY, Ordering::Release);
                
                offset += bytes_to_write;
                tile_idx = (tile_idx + 1) % ring_size;
            }
        })
    }
}
