use std::sync::{Arc, Mutex, OnceLock};
use std::alloc::{alloc_zeroed, dealloc, Layout};

pub static GLOBAL_CAPACITOR: OnceLock<Arc<GiantCapacitor>> = OnceLock::new();

pub fn get_capacitor() -> Arc<GiantCapacitor> {
    GLOBAL_CAPACITOR.get_or_init(|| {
        let avail_ram_gb = crate::sys_info::get_sys_info().ram_available_gb;
        let capacity_bytes = (avail_ram_gb * 1024.0 * 1024.0 * 1024.0 * 0.50) as usize; // Increased to 50% for high-performance Out-of-Core streaming
        GiantCapacitor::new_bytes(capacity_bytes)
    }).clone()
}

/// A giant RAM-based FIFO buffer ("The Capacitor") that sits between the SSD (io_uring)
/// and the CPU workers. It prefetches raw binary data to eliminate I/O wait times.
pub struct GiantCapacitor {
    ptr: *mut u8,
    layout: Layout,
    pub capacity: usize,
    /// Map ChunkID -> (offset_in_capacitor, size)
    chunks: Mutex<std::collections::HashMap<usize, (usize, usize)>>,
    current_write_pos: Mutex<usize>,
}

impl GiantCapacitor {
    pub fn new(capacity_mb: usize) -> Arc<Self> {
        Self::new_bytes(capacity_mb * 1024 * 1024)
    }

    pub fn new_bytes(capacity: usize) -> Arc<Self> {
        println!("[VNN] Initializing RAM Capacitor ({:.2} GB)...", capacity as f64 / 1024.0 / 1024.0 / 1024.0);
        
        // Ensure 4096-byte alignment for O_DIRECT
        let layout = Layout::from_size_align(capacity, 4096).expect("Invalid layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            panic!("[VNN] Failed to allocate {} bytes for RAM Capacitor", capacity);
        }

        Arc::new(Self {
            ptr,
            layout,
            capacity,
            chunks: Mutex::new(std::collections::HashMap::new()),
            current_write_pos: Mutex::new(0),
        })
    }

    /// Reserves a slice in the capacitor to write SSD data into.
    /// Implements a simple FIFO wrap-around logic.
    pub fn reserve_for_write(&self, size: usize) -> (usize, &mut [u8]) {
        let mut pos = self.current_write_pos.lock().unwrap();
        
        if *pos + size > self.capacity {
            // Wrap around (FIFO eviction)
            *pos = 0;
            self.chunks.lock().unwrap().clear(); 
        }

        let start = *pos;
        *pos += size;

        // Safety: We return a mutable slice to our internal buffer.
        // The caller (io_uring engine) will write directly here.
        let slice = unsafe {
             std::slice::from_raw_parts_mut(self.ptr.add(start), size)
        };

        (start, slice)
    }

    pub fn register_chunk(&self, chunk_id: usize, offset: usize, size: usize) {
        let mut chunks = self.chunks.lock().unwrap();
        chunks.insert(chunk_id, (offset, size));
    }

    pub fn get_chunk_ptr(&self, chunk_id: usize) -> Option<*const u8> {
        let chunks = self.chunks.lock().unwrap();
        if let Some(&(off, _)) = chunks.get(&chunk_id) {
            return Some(unsafe { self.ptr.add(off) as *const u8 });
        }
        None
    }
}
impl Drop for GiantCapacitor {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout); }
    }
}

unsafe impl Send for GiantCapacitor {}
unsafe impl Sync for GiantCapacitor {}
