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
#[repr(align(4096))]
pub struct StatefulTile {
    pub state: AtomicU32,
    pub tile_id: AtomicU32,
    _pad: [u8; 4088], // Pad to 4096 to ensure payload is aligned
    pub payload: UnsafeCell<[u8; 1048576]>, // Exactly 1MB
}

unsafe impl Sync for StatefulTile {}
unsafe impl Send for StatefulTile {}

impl StatefulTile {
    pub fn new() -> Self {
        Self {
            state: AtomicU32::new(TILE_EMPTY),
            tile_id: AtomicU32::new(0),
            _pad: [0; 4088],
            payload: UnsafeCell::new([0; 1048576]),
        }
    }
}

pub struct CrookScheduler {
    pub ring: Vec<Box<StatefulTile>>,
}

impl CrookScheduler {
    pub fn new(ring_size: usize) -> std::sync::Arc<Self> {
        let mut ring = Vec::with_capacity(ring_size);
        for _ in 0..ring_size {
            // Allocate on heap to avoid stack overflow for 1MB items
            ring.push(Box::new(StatefulTile::new()));
        }
        std::sync::Arc::new(Self { ring })
    }
    
    /// Starts the autonomous background worker analogous to MERA's peripheral dispatcher.
    pub fn start_io_worker(scheduler: std::sync::Arc<Self>, engine: std::sync::Arc<crate::io_uring_engine::DirectIoEngine>, total_bytes: u64) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut offset = 0;
            let mut tile_idx = 0;
            let ring_size = scheduler.ring.len();
            let mut current_id = 0;
            
            while offset < total_bytes {
                let tile = &scheduler.ring[tile_idx];

                // Tagged-Token architecture: Spin until tile is empty and ready to be reloaded
                while tile.state.compare_exchange(
                    TILE_EMPTY, 
                    TILE_READING_FROM_DISK, 
                    Ordering::Acquire, 
                    Ordering::Relaxed
                ).is_err() {
                    std::hint::spin_loop();
                }
                
                let bytes_to_read = std::cmp::min(1048576, total_bytes - offset); // max payload size (1MB)
                // We pad the read to a multiple of 4096 for O_DIRECT 
                let read_size = (bytes_to_read + 4095) & !4095;
                
                let payload_slice = unsafe { 
                    let ptr = tile.payload.get();
                    &mut (&mut *ptr)[0..std::cmp::min(1048576, read_size as usize)]
                };
                
                // Submit IO_URING read
                engine.read_chunk(offset, payload_slice);
                
                tile.tile_id.store(current_id, Ordering::Relaxed);
                
                // Mark tile as ready for CPU or GPU compute pipeline
                tile.state.store(TILE_READY_FOR_COMPUTE, Ordering::Release);
                
                offset += bytes_to_read;
                current_id += 1;
                tile_idx = (tile_idx + 1) % ring_size;
            }
        })
    }
    
    // Acquire an empty tile using Compare-and-Swap (CAS)
    pub fn acquire_empty(&self) -> Option<&StatefulTile> {
        for tile in &self.ring {
            if tile.state.compare_exchange(
                TILE_EMPTY, 
                TILE_READING_FROM_DISK, 
                Ordering::Acquire, 
                Ordering::Relaxed
            ).is_ok() {
                return Some(tile);
            }
        }
        None
    }
}
