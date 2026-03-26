use io_uring::{opcode, IoUring, types};
use std::os::unix::io::AsRawFd;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// A 1MB aligned buffer for ZFS / O_DIRECT compatibility
pub struct AlignedBuffer {
    pub ptr: *mut u8,
    pub layout: Layout,
    pub size: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocates a new zeroed buffer aligned strictly to 1MB, padded to 4096 bytes for O_DIRECT safely.
    pub fn new(size: usize) -> Self {
        // Enforce 1MB alignment for optimal ZFS Recordsize interactions
        let align = 1048576; 
        // Pad size to multiple of 4096 for O_DIRECT requirement
        let padded_size = (size + 4095) & !4095;
        let layout = Layout::from_size_align(padded_size, align).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            panic!("Failed to allocate 1MB aligned buffer of size {}", padded_size);
        }
        Self { ptr, layout, size: padded_size } // We track the padded size
    }

    /// Returns the buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
    
    /// Returns the buffer as a read-only slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout); }
    }
}

/// Extreme out-of-core file engine bypassing Linux VFS Page Cache
pub struct DirectIoEngine {
    ring: Mutex<IoUring>,
    pub file: Arc<File>,
    /// Multiplex results for multiple threads sharing the same ring
    pending_completions: Mutex<HashMap<u64, i32>>,
}

unsafe impl Send for DirectIoEngine {}
unsafe impl Sync for DirectIoEngine {}

impl DirectIoEngine {
    /// Initializes a new Direct I/O engine bypassing the VFS page cache.
    pub fn new(path: &str, read_only: bool) -> Self {
        let mut options = OpenOptions::new();
        options.read(true);
        if !read_only {
            options.write(true).create(true).truncate(true);
        }
        // Use O_DIRECT to bypass page cache (Software DMA)
        options.custom_flags(libc::O_DIRECT);
        
        let file = options.open(path).expect("Failed to open file with O_DIRECT. Ensure filesystem supports it.");
        
        // 128 queue depth
        let ring = IoUring::new(128).expect("Failed to init io_uring");
        Self { 
            ring: Mutex::new(ring), 
            file: Arc::new(file), 
            pending_completions: Mutex::new(HashMap::new()),
        }
    }

    /// Helper to wait for a specific completion ID while parking others
    fn wait_for_id(&self, target_id: u64) -> i32 {
        loop {
            // 1. Check if it's already in the pending map
            {
                let mut pending = self.pending_completions.lock().unwrap();
                if let Some(res) = pending.remove(&target_id) {
                    return res;
                }
            }

            // 2. Otherwise, pull from the ring
            let mut ring = self.ring.lock().unwrap();
            
            // Check again inside the lock (someone might have pulled it while we were getting the lock)
            {
                let mut pending = self.pending_completions.lock().unwrap();
                if let Some(res) = pending.remove(&target_id) {
                    return res;
                }
            }

            // block for at least 1 result
            ring.submit_and_wait(1).expect("io_uring submit_and_wait failed");
            
            let mut cq = ring.completion();
            while let Some(cqe) = cq.next() {
                let id = cqe.user_data();
                let res = cqe.result();
                
                if id == target_id {
                    return res;
                } else {
                    // Park it for others
                    self.pending_completions.lock().unwrap().insert(id, res);
                }
            }
        }
    }

    /// Read an aligned block directly into the buffer
    pub fn read_chunk(&self, offset: u64, buffer: &mut [u8]) {
        let id = 0x8000_0000_0000_0111u64; // High bit set for Sync Read
        let fd = types::Fd(self.file.as_raw_fd());
        let read_e = opcode::Read::new(fd, buffer.as_mut_ptr(), buffer.len() as _)
            .offset(offset as _)
            .build()
            .user_data(id);
        
        {
            let mut ring = self.ring.lock().unwrap();
            unsafe {
                ring.submission().push(&read_e).expect("submission queue is full");
            }
            ring.submit().expect("io_uring submit failed");
        }

        let res = self.wait_for_id(id);

        if res < 0 {
            let err = std::io::Error::from_raw_os_error(-res);
            panic!("io_uring read failed: {} (offset={}, len={})", err, offset, buffer.len());
        }
    }

    /// Submits a read request to the capacitor without waiting.
    /// Returns the capacitor offset used for this write.
    pub fn submit_read_to_capacitor(&self, offset: u64, size: usize, capacitor: &crate::tensor::GiantCapacitor, chunk_id: usize) -> usize {
        let (cap_offset, slice) = capacitor.reserve_for_write(size);
        let fd = types::Fd(self.file.as_raw_fd());
        
        let read_e = opcode::Read::new(fd, slice.as_mut_ptr(), slice.len() as _)
            .offset(offset as _)
            .build()
            .user_data(chunk_id as u64); // user_data = chunk_id
        
        let mut ring = self.ring.lock().unwrap();
        unsafe {
            ring.submission().push(&read_e).expect("submission queue is full");
        }
        ring.submit().expect("io_uring submit failed");
        
        cap_offset
    }

    /// Pulls all available completions and registers them with the capacitor.
    /// This should be called periodically by the prefetcher thread.
    pub fn poll_completions(&self, capacitor: &crate::tensor::GiantCapacitor, pending_map: &mut HashMap<usize, (usize, usize)>) {
        // 1. Process what's already parked in pending_completions
        {
            let mut pending = self.pending_completions.lock().unwrap();
            let keys: Vec<u64> = pending.keys().cloned().collect();
            for id in keys {
                let chunk_id = id as usize;
                if pending_map.contains_key(&chunk_id) {
                    let res = pending.remove(&id).unwrap();
                    if res < 0 {
                         let err = std::io::Error::from_raw_os_error(-res);
                         panic!("io_uring async read (parked) failed: {} (chunk_id={})", err, chunk_id);
                    }
                    let (cap_offset, size) = pending_map.remove(&chunk_id).unwrap();
                    capacitor.register_chunk(chunk_id, cap_offset, size);
                }
            }
        }

        // 2. Pull fresh from the ring
        let mut ring = self.ring.lock().unwrap();
        let mut cq = ring.completion();
        
        while let Some(cqe) = cq.next() {
            let id = cqe.user_data();
            let res = cqe.result();
            let chunk_id = id as usize;
            
            if res < 0 {
                let err = std::io::Error::from_raw_os_error(-res);
                panic!("io_uring async read failed: {} (chunk_id={})", err, chunk_id);
            }
            
            if let Some((cap_offset, size)) = pending_map.remove(&chunk_id) {
                capacitor.register_chunk(chunk_id, cap_offset, size);
            } else {
                // Not for us? Park it for others
                self.pending_completions.lock().unwrap().insert(id, res);
            }
        }
    }
    
    /// Write an aligned block directly from the buffer
    pub fn write_chunk(&self, offset: u64, buffer: &[u8]) {
        let id = 0x8000_0000_0000_0222u64; // High bit set for Sync Write
        let fd = types::Fd(self.file.as_raw_fd());
        let write_e = opcode::Write::new(fd, buffer.as_ptr(), buffer.len() as _)
            .offset(offset as _)
            .build()
            .user_data(id);
            
        {
            let mut ring = self.ring.lock().unwrap();
            unsafe {
                ring.submission().push(&write_e).expect("submission queue is full");
            }
            ring.submit().expect("io_uring submit failed");
        }

        let res = self.wait_for_id(id);

        if res < 0 {
            let err = std::io::Error::from_raw_os_error(-res);
            panic!("io_uring write failed: {} (offset={}, len={})", err, offset, buffer.len());
        }
    }
}
