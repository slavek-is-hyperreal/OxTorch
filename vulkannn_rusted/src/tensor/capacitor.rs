use std::sync::{Arc, Mutex, RwLock, OnceLock};
use std::alloc::{dealloc, Layout};

pub static GLOBAL_CAPACITOR: OnceLock<RwLock<Option<Arc<GiantCapacitor>>>> = OnceLock::new();

pub fn get_capacitor() -> Arc<GiantCapacitor> {
    let lock = GLOBAL_CAPACITOR.get_or_init(|| RwLock::new(None));
    let info = crate::sys_info::get_sys_info();
    
    // 1. Sprawdź czy już mamy optymalny bufor
    {
        let read = lock.read().unwrap();
        if let Some(cap) = read.as_ref() {
            // PULA = to co system pokazuje jako wolne + to co MY już zajmujemy
            let os_avail_gb = info.ram_available_gb;
            let current_cap_gb = cap.capacity as f64 / 1024.0 / 1024.0 / 1024.0;
            let total_pool_gb = os_avail_gb + current_cap_gb;
            let target_gb = total_pool_gb * 0.50;
            
            // Jeśli obecny bufor jest blisko docelowego (np. > 80%), nie ruszaj go
            if current_cap_gb >= (target_gb * 0.8) {
                return cap.clone();
            }
            println!("[VNN] True Adaptive: RAM pool increased from {:.2} GB to {:.2} GB. Resizing...", current_cap_gb, target_gb);
        }
    }

    // 2. Inicjalizacja lub zmiana rozmiaru (Write Lock)
    let mut write = lock.write().unwrap();
    
    // Re-check pod blokadą, czy inna nić nas nie ubiegła
    if let Some(cap) = write.as_ref() {
        let os_avail_gb = info.ram_available_gb;
        let current_cap_gb = cap.capacity as f64 / 1024.0 / 1024.0 / 1024.0;
        let total_pool_gb = os_avail_gb + current_cap_gb;
        let target_gb = total_pool_gb * 0.50;
        if current_cap_gb >= (target_gb * 0.8) {
             return cap.clone();
        }
    }

    let os_avail_gb = info.ram_available_gb;
    let floor_bytes = crate::hardware_config::CAPACITOR_FLOOR_MB * 1024 * 1024;
    
    // Na starcie (gdy current_cap_gb = 0) bierzemy po prostu 50% dostępnego
    let target_bytes = (os_avail_gb * 1024.0 * 1024.0 * 1024.0 * 0.50) as usize;
    let final_capacity = target_bytes.max(floor_bytes);
    
    let new_cap = GiantCapacitor::new_bytes(final_capacity);
    *write = Some(new_cap.clone());
    new_cap
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
        println!("[VNN] True Adaptive: Reserving physical RAM pages...");
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            println!("[VNN] CRITICAL: RAM Capacitor allocation failed! Out of Memory.");
            panic!("[VNN] Failed to allocate {} bytes for RAM Capacitor", capacity);
        }
        println!("[VNN] RAM Capacitor successfully mapped at {:?}.", ptr);

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
