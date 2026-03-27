use ash::vk;
use crate::tensor::DataType;
use crate::vulkan::memory::CachedBuffer;
use crate::cpu::{convert_f16_to_f32, convert_bf16_to_f32, convert_f32_to_f16, convert_f32_to_bf16};

pub mod elementwise;
pub mod index_select;
pub mod activation;
pub mod reduce;
pub mod softmax;
pub mod matmul;
pub mod bit_linear;
pub mod norm;

pub fn begin_cmd(device: &ash::Device, pool: vk::CommandPool) -> vk::CommandBuffer {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];
    let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info) }.unwrap();
    cmd
}

pub fn upload_to_stage(src_raw: &[u8], stage: &CachedBuffer, dtype: DataType) {
    let ptr = stage.mapped_ptr.unwrap() as *mut f32;
    let bytes_per_elem = match dtype {
        DataType::F32  => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let num_elements = src_raw.len() / bytes_per_elem;
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(ptr, num_elements) };

    if dtype == DataType::F16 {
        let src_slice = bytemuck::cast_slice::<u8, half::f16>(src_raw);
        convert_f16_to_f32(src_slice, dst_slice);
    } else if dtype == DataType::BF16 {
        let src_slice = bytemuck::cast_slice::<u8, half::bf16>(src_raw);
        convert_bf16_to_f32(src_slice, dst_slice);
    } else if dtype == DataType::Int8 {
        let src_slice = bytemuck::cast_slice::<u8, i8>(src_raw);
        for i in 0..num_elements {
            dst_slice[i] = src_slice[i] as f32;
        }
    } else {
        unsafe { std::ptr::copy_nonoverlapping(src_raw.as_ptr(), ptr as *mut u8, src_raw.len()); }
    }
}

pub fn download_from_stage(dst_raw: &mut [u8], stage: &CachedBuffer, dtype: DataType) {
    let ptr = stage.mapped_ptr.unwrap() as *const f32;
    let bytes_per_elem = match dtype {
        DataType::F32  => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let num_elements = dst_raw.len() / bytes_per_elem;
    let src_slice = unsafe { std::slice::from_raw_parts(ptr, num_elements) };

    if dtype == DataType::F16 {
        let dst_slice = bytemuck::cast_slice_mut::<u8, half::f16>(dst_raw);
        convert_f32_to_f16(src_slice, dst_slice);
    } else if dtype == DataType::BF16 {
        let dst_slice = bytemuck::cast_slice_mut::<u8, half::bf16>(dst_raw);
        convert_f32_to_bf16(src_slice, dst_slice);
    } else if dtype == DataType::Int8 {
        let dst_slice = bytemuck::cast_slice_mut::<u8, i8>(dst_raw);
        for i in 0..num_elements {
            dst_slice[i] = src_slice[i] as i8;
        }
    } else {
        unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, dst_raw.as_mut_ptr(), dst_raw.len()); }
    }
}

pub fn upload_to_stage_raw(src_raw: &[u8], stage: &CachedBuffer) {
    let ptr = stage.mapped_ptr.expect("Stage buffer must be mapped");
    unsafe { std::ptr::copy_nonoverlapping(src_raw.as_ptr(), ptr, src_raw.len()); }
}

pub fn download_from_stage_raw(dst_raw: &mut [u8], stage: &CachedBuffer) {
    let ptr = stage.mapped_ptr.expect("Stage buffer must be mapped");
    unsafe { std::ptr::copy_nonoverlapping(ptr, dst_raw.as_mut_ptr(), dst_raw.len()); }
}
