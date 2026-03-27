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

/// Represents a single I/O slot (either Source A, Source B, or Result).
/// Can point to either a local AlignedBuffer or a Zero-Copy Capacitor address.
pub struct TileSlot {
    pub local_buf: UnsafeCell<crate::io_uring_engine::AlignedBuffer>,
    pub capacitor_ptr: AtomicPtr<u8>,
}

impl TileSlot {
    pub fn new(size: usize) -> Self {
        Self {
            local_buf: UnsafeCell::new(crate::io_uring_engine::AlignedBuffer::new(size)),
            capacitor_ptr: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    pub fn get_ptr(&self, size: usize) -> *mut u8 {
        let cap = self.capacitor_ptr.load(Ordering::Acquire);
        if !cap.is_null() {
            cap
        } else {
            unsafe { (*self.local_buf.get()).as_mut_slice()[..size].as_mut_ptr() }
        }
    }
}

/// MERA-400 CROOK OS Inspired Stateful Tile
/// Enhanced with Multi-Stream Bitmask Barrier.
pub struct StatefulTile {
    pub state: AtomicU32,
    pub ready_bits: AtomicU32, // Bit 0: Arg A, Bit 1: Arg B, Bit 2: Arg C...
    pub tile_id: AtomicU32,
    pub slot_a: TileSlot,
    pub slot_b: TileSlot,
    pub slot_res: TileSlot,
}

unsafe impl Sync for StatefulTile {}
unsafe impl Send for StatefulTile {}

impl StatefulTile {
    /// Constructs a clear, EMPTY Stateful Tile with multi-slots.
    pub fn new(size: usize) -> Self {
        Self {
            state: AtomicU32::new(TILE_EMPTY),
            ready_bits: AtomicU32::new(0),
            tile_id: AtomicU32::new(0),
            slot_a: TileSlot::new(size),
            slot_b: TileSlot::new(size),
            slot_res: TileSlot::new(size),
        }
    }

    /// Resets the tile for a new operation.
    pub fn reset(&self) {
        self.state.store(TILE_EMPTY, Ordering::Release);
        self.ready_bits.store(0, Ordering::Release);
        self.slot_a.capacitor_ptr.store(std::ptr::null_mut(), Ordering::Release);
        self.slot_b.capacitor_ptr.store(std::ptr::null_mut(), Ordering::Release);
        self.slot_res.capacitor_ptr.store(std::ptr::null_mut(), Ordering::Release);
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
    /// `source_idx`: 0 for A, 1 for B, etc.
    pub fn start_read_worker(scheduler: std::sync::Arc<Self>, engine: std::sync::Arc<crate::io_uring_engine::DirectIoEngine>, total_bytes: u64, source_idx: u32) -> std::thread::JoinHandle<()> {
        let bit = 1 << source_idx;
        std::thread::spawn(move || {
            let mut offset = 0;
            let mut tile_idx = 0;
            let ring_size = scheduler.ring.len();
            let mut current_id = 0;
            
            while offset < total_bytes {
                let tile = &scheduler.ring[tile_idx];

                // For the FIRST source (A), we wait for TILE_EMPTY and set to READING.
                // For subsequent sources (B), we just wait until the slot is free (state <= READING).
                if source_idx == 0 {
                    while tile.state.compare_exchange(
                        TILE_EMPTY, 
                        TILE_READING_FROM_DISK, 
                        Ordering::Acquire, 
                        Ordering::Relaxed
                    ).is_err() {
                        std::hint::spin_loop();
                    }
                } else {
                    // Subsequent sources wait for the tile to be in READING state (initiated by source 0)
                    while tile.state.load(Ordering::Acquire) != TILE_READING_FROM_DISK {
                        std::hint::spin_loop();
                    }
                }
                
                let bytes_to_read = std::cmp::min(scheduler.tile_size as u64, total_bytes - offset);
                let slot = if source_idx == 0 { &tile.slot_a } else { &tile.slot_b };

                // Option A: Zero-Copy via Capacitor
                let mut read_done = false;
                if let Some(cap) = &scheduler.capacitor {
                   if let Some(ptr) = cap.get_chunk_ptr(current_id as usize) {
                       slot.capacitor_ptr.store(ptr as *mut u8, Ordering::Release);
                       read_done = true;
                   }
                }

                if !read_done {
                    // Option B: Standard Read-through
                    slot.capacitor_ptr.store(std::ptr::null_mut(), Ordering::Release);
                    let payload_slice = unsafe { 
                       let buf = &mut *slot.local_buf.get();
                       &mut buf.as_mut_slice()[0..bytes_to_read as usize]
                    };
                    engine.read_chunk(offset, payload_slice);
                }

                // IMPORTANT: Bitmask Barrier. 
                // We add our bit. If result is correctly masked, we might be the last one.
                tile.ready_bits.fetch_or(bit, Ordering::Release);
                
                // If we are source 0, we manage the global tile ID
                if source_idx == 0 {
                    tile.tile_id.store(current_id, Ordering::Relaxed);
                }

                // Check if ALL expected bits are set... 
                // (Wait, this is handled by the Compute Worker in our revised plan)
                // Once all sources are ready (checked by the loop), progress to READY_FOR_COMPUTE.
                // For simplicity, for unary ops (bits == 1), it works instantly.
                // For binary ops (bits == 3), the LAST reader will trigger the compute state.
                // BUT better: let the CPU worker spin on ready_bits.

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
                // Flush from the RESULT slot
                let payload_ptr = tile.slot_res.get_ptr(bytes_to_write as usize);
                let payload_slice = unsafe { std::slice::from_raw_parts(payload_ptr, bytes_to_write as usize) };
                
                engine.write_chunk(offset, payload_slice);
                
                // Mark as EMPTY for the read workers to reuse
                tile.reset();
                
                offset += bytes_to_write;
                tile_idx = (tile_idx + 1) % ring_size;
            }
        })
    }
}
