use io_uring::{opcode, IoUring, types};
use std::os::unix::io::AsRawFd;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::{Arc, Mutex};

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
}

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
        Self { ring: Mutex::new(ring), file: Arc::new(file) }
    }

    /// Read an aligned block directly into the buffer
    pub fn read_chunk(&self, offset: u64, buffer: &mut [u8]) {
        let fd = types::Fd(self.file.as_raw_fd());
        let read_e = opcode::Read::new(fd, buffer.as_mut_ptr(), buffer.len() as _)
            .offset(offset as _)
            .build()
            .user_data(1);
        
        let res = {
            let mut ring = self.ring.lock().unwrap();
            unsafe {
                ring.submission().push(&read_e).expect("submission queue is full");
            }
            ring.submit_and_wait(1).expect("submit_and_wait failed");
            
            let cqe = ring.completion().next().expect("completion queue is empty");
            assert_eq!(cqe.user_data(), 1);
            cqe.result()
        };

        if res < 0 {
            let err = std::io::Error::from_raw_os_error(-res);
            panic!("io_uring read failed: {} (offset={}, len={})", err, offset, buffer.len());
        }
    }
    
    /// Write an aligned block directly from the buffer
    pub fn write_chunk(&self, offset: u64, buffer: &[u8]) {
        let fd = types::Fd(self.file.as_raw_fd());
        let write_e = opcode::Write::new(fd, buffer.as_ptr(), buffer.len() as _)
            .offset(offset as _)
            .build()
            .user_data(2);
            
        let res = {
            let mut ring = self.ring.lock().unwrap();
            unsafe {
                ring.submission().push(&write_e).expect("submission queue is full");
            }
            ring.submit_and_wait(1).expect("submit_and_wait failed");
            
            let cqe = ring.completion().next().expect("completion queue is empty");
            assert_eq!(cqe.user_data(), 2);
            cqe.result()
        };

        if res < 0 {
            let err = std::io::Error::from_raw_os_error(-res);
            panic!("io_uring write failed: {} (offset={}, len={})", err, offset, buffer.len());
        }
    }
}
