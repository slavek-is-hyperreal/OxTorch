use std::fs::File;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use memmap2::{Mmap, MmapOptions};

/// The L3 Cache: Raw SSD storage mapped to memory.
#[allow(dead_code)]
pub struct L3Cache;

impl L3Cache {
    #[allow(dead_code)]
    pub fn map_ssd_tensor(path: &str) -> std::io::Result<Arc<Mmap>> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // --- EXTREME PERFORMANCE: POSIX MADV_WILLNEED + MADV_SEQUENTIAL ---
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_WILLNEED);
            libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_SEQUENTIAL);
        }

        Ok(Arc::new(mmap))
    }
}

/// Budget Trackers
pub struct MemoryBudgets {
    #[allow(dead_code)]
    pub l1_vram_max_bytes: usize,
    #[allow(dead_code)]
    pub l2_ram_max_bytes: usize,
    
    #[allow(dead_code)]
    pub l1_vram_used: usize,
    #[allow(dead_code)]
    pub l2_ram_used: usize,
}

/// Detects available RAM on Linux (MemAvailable from /proc/meminfo)
pub fn get_available_ram() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            // Leave 2GB safety margin for system/OS/Antigravity
                            let bytes = kb * 1024;
                            return bytes.saturating_sub(2 * 1024 * 1024 * 1024);
                        }
                    }
                }
            }
        }
    }
    // Fallback for non-Linux or failures: provide a conservative 4GB
    4 * 1024 * 1024 * 1024
}

pub static BUDGETS: OnceLock<Mutex<MemoryBudgets>> = OnceLock::new();

pub fn init_budgets() {
    BUDGETS.get_or_init(|| {
        let avail_ram = get_available_ram();
        println!("[vulkannn_rusted] Detected Available RAM for Compute: {:.2} GB", avail_ram as f64 / 1024.0 / 1024.0 / 1024.0);
        
        Mutex::new(MemoryBudgets {
            l1_vram_max_bytes: 1024 * 1024 * 1024, // 1GB MVP VRAM
            l2_ram_max_bytes: avail_ram,
            l1_vram_used: 0,
            l2_ram_used: 0,
        })
    });
}

/// A background prefetching engine moving data from L3 (SSD) -> L2 (RAM)
pub struct PrefetchEngine {
    #[allow(dead_code)]
    tx: std::sync::mpsc::Sender<PrefetchRequest>,
}

pub struct PrefetchRequest {
    pub mmap: Arc<Mmap>,
    pub signal_done: Option<std::sync::mpsc::Sender<()>>,
}

pub static PREFETCHER: OnceLock<PrefetchEngine> = OnceLock::new();

pub fn init_prefetcher() {
    PREFETCHER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<PrefetchRequest>();
        
        thread::spawn(move || {
            for request in rx {
                // Background worker: touch pages to trigger OS readahead
                let ptr = request.mmap.as_ptr();
                let len = request.mmap.len();
                
                // We touch one byte every 64KB (a typical readahead cluster size)
                let mut i = 0;
                while i < len {
                    unsafe {
                        let _unused = std::ptr::read_volatile(ptr.add(i));
                    }
                    i += 65536; 
                }

                if let Some(sig) = request.signal_done {
                    let _ = sig.send(());
                }
            }
        });

        PrefetchEngine { tx }
    });
}

#[allow(dead_code)]
pub fn prefetch_tensor(mmap: Arc<Mmap>) {
    if let Some(engine) = PREFETCHER.get() {
        let _ = engine.tx.send(PrefetchRequest {
            mmap,
            signal_done: None,
        });
    }
}
