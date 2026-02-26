use std::fs::File;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::collections::VecDeque;
use memmap2::{Mmap, MmapOptions};

/// The L3 Cache: Raw SSD storage mapped to memory.
pub struct L3Cache;

impl L3Cache {
    pub fn map_ssd_tensor(path: &str) -> std::io::Result<Arc<Mmap>> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // --- EXTREME PERFORMANCE: POSIX MADV_WILLNEED ---
        // Telling the Linux kernel to start paging this immediately into OS RAM
        // before the GPU actually requests it.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_WILLNEED,
            );
        }

        Ok(Arc::new(mmap))
    }
}

/// Budget Trackers
pub struct MemoryBudgets {
    pub l1_vram_max_bytes: usize,
    pub l2_ram_max_bytes: usize,
    
    pub l1_vram_used: usize,
    pub l2_ram_used: usize,
}

pub static BUDGETS: OnceLock<Mutex<MemoryBudgets>> = OnceLock::new();

pub fn init_budgets() {
    BUDGETS.get_or_init(|| {
        Mutex::new(MemoryBudgets {
            l1_vram_max_bytes: 1024 * 1024 * 1024, // 1GB MVP VRAM
            l2_ram_max_bytes: 8 * 1024 * 1024 * 1024, // 8GB MVP RAM
            l1_vram_used: 0,
            l2_ram_used: 0,
        })
    });
}

/// A background prefetching engine moving data from L3 (SSD) -> L2 (RAM) -> L1 (VRAM)
pub struct PrefetchEngine {
    tx: std::sync::mpsc::Sender<PrefetchRequest>,
}

pub struct PrefetchRequest {
    pub file_path: String,
    pub shape: Vec<usize>,
    pub callback: Box<dyn FnOnce(Vec<f32>) + Send + 'static>,
}

pub static PREFETCHER: OnceLock<PrefetchEngine> = OnceLock::new();

pub fn init_prefetcher() {
    PREFETCHER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<PrefetchRequest>();
        
        thread::spawn(move || {
            for request in rx {
                // Background worker: Load from L3 (SSD) into L2 (RAM vector)
                if let Ok(mmap) = L3Cache::map_ssd_tensor(&request.file_path) {
                    let ptr = mmap.as_ptr() as *const f32;
                    let len = mmap.len() / 4;
                    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                    
                    // Copy to RAM (L2) - in reality we might zero-copy this until DMA,
                    // but for now, we simulate forcing it into active L2 RAM pages.
                    let l2_data = slice.to_vec();
                    
                    // Send back to caller
                    (request.callback)(l2_data);
                } else {
                    eprintln!("Failed to prefetch {}", request.file_path);
                }
            }
        });

        PrefetchEngine { tx }
    });
}
