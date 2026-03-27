use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use gpu_allocator::MemoryLocation;
use crate::vulkan::context::BACKEND;

#[derive(Clone, Copy, Debug)]
pub struct PoolBlock {
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    pub used: bool,
}

/// A reusable Vulkan buffer paired with its native memory allocation.
/// Tracked by the memory allocator to enable zero-copy VRAM reuse.
pub struct CachedBuffer {
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub cpu_visible: bool,
    pub mapped_ptr: Option<*mut u8>,
    pub pool_offset: Option<vk::DeviceSize>, // New: Tracks offset if sub-allocated
}

unsafe impl Send for CachedBuffer {}
unsafe impl Sync for CachedBuffer {}

impl CachedBuffer {
    pub fn copy_for_async(&self) -> Self {
        Self {
            size: self.size,
            usage: self.usage,
            buffer: self.buffer,
            allocation: None, // Do NOT copy allocation
            cpu_visible: self.cpu_visible,
            mapped_ptr: self.mapped_ptr,
            pool_offset: self.pool_offset,
        }
    }
}

/// Requests a Vulkan memory allocation.
/// Attempts to fetch a suitable existing block from the shared `buffer_cache`.
pub fn get_buffer(size: vk::DeviceSize, usage: vk::BufferUsageFlags, label: Option<&str>, cpu_visible: bool) -> CachedBuffer {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    
    let alignment: vk::DeviceSize = 256;
    let aligned_size = (size + alignment - 1) & !(alignment - 1);

    if !cpu_visible {
        let mut free_list = backend.pool_free_list.lock().unwrap();
        if let Some(idx) = free_list.iter().position(|b| !b.used && b.size >= aligned_size) {
            let block = free_list[idx];
            let remaining = block.size - aligned_size;
            
            free_list[idx].used = true;
            free_list[idx].size = aligned_size;
            
            let offset = block.offset;
            
            if remaining > 0 {
                free_list.insert(idx + 1, PoolBlock {
                    offset: offset + aligned_size,
                    size: remaining,
                    used: false,
                });
            }
            
            return CachedBuffer {
                size: aligned_size,
                usage,
                buffer: backend.pool_buffer,
                allocation: None,
                cpu_visible: false,
                mapped_ptr: None,
                pool_offset: Some(offset),
            };
        }
    }

    if let Ok(mut cache) = backend.buffer_cache.lock() {
        if let Some(idx) = cache.iter().position(|b| b.size >= size && b.usage.contains(usage) && b.cpu_visible == cpu_visible && b.pool_offset.is_none()) {
            let cached = cache.swap_remove(idx);
            return cached;
        }
    }
    
    let buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { backend.device.create_buffer(&buffer_info, None) }.unwrap();
    
    let requirements = unsafe { backend.device.get_buffer_memory_requirements(buffer) };
    
    let location = if cpu_visible { 
        MemoryLocation::CpuToGpu
    } else { 
        MemoryLocation::GpuOnly 
    };
    
    let mut retry_count = 0;
    let allocation = loop {
        let result = backend.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: label.unwrap_or("Buffer"),
            requirements,
            location,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        });

        match result {
            Ok(alloc) => break alloc,
            Err(e) => {
                if retry_count == 0 {
                    println!("[VNN] VRAM Allocation failed: {:?}. Clearing cache...", e);
                    clear_all_caches();
                    retry_count += 1;
                    continue;
                } else {
                    panic!("[VNN] CRITICAL: VRAM Allocation failed: {:?}", e);
                }
            }
        }
    };
    
    let cpu_visible_actual = allocation.mapped_ptr().is_some();
    let mapped_ptr = allocation.mapped_ptr().map(|p| p.as_ptr() as *mut u8);
    
    unsafe { backend.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }.unwrap();
    
    CachedBuffer { size, usage, buffer, allocation: Some(allocation), cpu_visible: cpu_visible_actual, mapped_ptr, pool_offset: None }
}

pub fn get_buffer_readback(size: vk::DeviceSize, usage: vk::BufferUsageFlags, label: Option<&str>) -> CachedBuffer {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");

    if let Ok(mut cache) = backend.buffer_cache.lock() {
        if let Some(idx) = cache.iter().position(|b| b.size >= size && b.usage.contains(usage) && b.cpu_visible && b.pool_offset.is_none()) {
            let cached = cache.swap_remove(idx);
            return cached;
        }
    }

    let buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { backend.device.create_buffer(&buffer_info, None) }.unwrap();
    let requirements = unsafe { backend.device.get_buffer_memory_requirements(buffer) };

    let mut retry_count = 0;
    let allocation = loop {
        let result = backend.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: label.unwrap_or("ReadbackBuffer"),
            requirements,
            location: MemoryLocation::GpuToCpu,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        });
        match result {
            Ok(alloc) => break alloc,
            Err(e) => {
                if retry_count == 0 { clear_all_caches(); retry_count += 1; }
                else { panic!("[VNN] Readback alloc failed: {:?}", e); }
            }
        }
    };

    let mapped_ptr = allocation.mapped_ptr().map(|p| p.as_ptr() as *mut u8);
    unsafe { backend.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }.unwrap();
    CachedBuffer { size, usage, buffer, allocation: Some(allocation), cpu_visible: true, mapped_ptr, pool_offset: None }
}

pub fn recycle_buffer(cached: CachedBuffer) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    
    if let Some(offset) = cached.pool_offset {
        let mut free_list = backend.pool_free_list.lock().unwrap();
        if let Some(idx) = free_list.iter().position(|b| b.offset == offset && b.used) {
            free_list[idx].used = false;
            coalesce_pool(idx, &mut free_list);
            return;
        }
    }

    if let Ok(mut cache) = backend.buffer_cache.lock() {
        let current_total: u64 = cache.iter().map(|b| b.size).sum();
        if current_total > 512 * 1024 * 1024 {
            prune_buffer_cache(2); 
        }
        cache.push(cached);
    }
}

fn coalesce_pool(idx: usize, free_list: &mut Vec<PoolBlock>) {
    if idx + 1 < free_list.len() && !free_list[idx + 1].used {
        let next = free_list.remove(idx + 1);
        free_list[idx].size += next.size;
    }
    if idx > 0 && !free_list[idx - 1].used {
        let current = free_list.remove(idx);
        free_list[idx - 1].size += current.size;
    }
}

pub fn prune_buffer_cache(count: usize) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        let to_remove = count.min(cache.len());
        for _ in 0..to_remove {
            if let Some(buf) = cache.pop() {
                destroy_cached_buffer(buf);
            }
        }
    }
}

fn destroy_cached_buffer(mut buf: CachedBuffer) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    unsafe {
        backend.device.destroy_buffer(buf.buffer, None);
        if let Some(alloc) = buf.allocation.take() {
            backend.allocator.lock().unwrap().free(alloc).unwrap();
        }
    }
}

pub fn clear_all_caches() {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    if let Ok(mut cache) = backend.buffer_cache.lock() {
        while !cache.is_empty() {
            let buf = cache.pop().unwrap();
            destroy_cached_buffer(buf);
        }
    }
}
