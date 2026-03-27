use ash::vk;
use std::sync::atomic::Ordering;
use crate::tensor::DataType;
use crate::vulkan::context::{BACKEND, AsyncOp, poll_async_ops};
use crate::vulkan::memory::{get_buffer, get_buffer_readback};
use crate::vulkan::ops::{begin_cmd, upload_to_stage_raw, download_from_stage_raw};

pub fn execute_bit_linear_into(a_raw: &[u8], b_raw: &[u8], s_raw: &[u8], bias_raw: &[u8], out_raw: &mut [u8], m: u32, k: u32, n: u32, dtype: DataType) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let num_bytes_a = (m * k) as vk::DeviceSize;
    let num_bytes_b = (n * k) as vk::DeviceSize;
    let num_bytes_s = (n * 4) as vk::DeviceSize;
    let num_bytes_bias = (n * 4) as vk::DeviceSize;
    let num_bytes_out = (m * n * 4) as vk::DeviceSize;

    let buf_a = get_buffer(num_bytes_a, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_A"), false);
    let buf_b = get_buffer(num_bytes_b, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_B"), false);
    let buf_s = get_buffer(num_bytes_s, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_S"), false);
    let buf_bias = get_buffer(num_bytes_bias, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_Bias"), false);
    let buf_out = get_buffer(num_bytes_out, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_Out"), false);

    let stage_a = get_buffer(num_bytes_a, vk::BufferUsageFlags::TRANSFER_SRC, Some("BitLinear_Stage_A"), true);
    let stage_b = get_buffer(num_bytes_b, vk::BufferUsageFlags::TRANSFER_SRC, Some("BitLinear_Stage_B"), true);
    let stage_s = get_buffer(num_bytes_s, vk::BufferUsageFlags::TRANSFER_SRC, Some("BitLinear_Stage_S"), true);
    let stage_bias = get_buffer(num_bytes_bias, vk::BufferUsageFlags::TRANSFER_SRC, Some("BitLinear_Stage_Bias"), true);
    let stage_out = get_buffer_readback(num_bytes_out, vk::BufferUsageFlags::TRANSFER_DST, Some("BitLinear_Stage_Out"));

    upload_to_stage_raw(a_raw, &stage_a);
    upload_to_stage_raw(b_raw, &stage_b);
    upload_to_stage_raw(s_raw, &stage_s);
    if bias_raw.is_empty() {
        let ptr = stage_bias.mapped_ptr.unwrap() as *mut f32;
        unsafe { std::slice::from_raw_parts_mut(ptr, n as usize).fill(0.0); }
    } else {
        upload_to_stage_raw(bias_raw, &stage_bias);
    }

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_a.pool_offset.unwrap_or(0), size: num_bytes_a }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_b.pool_offset.unwrap_or(0), size: num_bytes_b }]);
        backend.device.cmd_copy_buffer(cmd, stage_s.buffer, buf_s.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_s.pool_offset.unwrap_or(0), size: num_bytes_s }]);
        backend.device.cmd_copy_buffer(cmd, stage_bias.buffer, buf_bias.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_bias.pool_offset.unwrap_or(0), size: num_bytes_bias }]);

        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).size(buf_a.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).size(buf_b.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_s.buffer).offset(buf_s.pool_offset.unwrap_or(0)).size(buf_s.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_bias.buffer).offset(buf_bias.pool_offset.unwrap_or(0)).size(buf_bias.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_bit_linear.lock().unwrap().next();
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).range(buf_a.size)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).range(buf_b.size)];
        let info_s = [vk::DescriptorBufferInfo::default().buffer(buf_s.buffer).offset(buf_s.pool_offset.unwrap_or(0)).range(buf_s.size)];
        let info_bias = [vk::DescriptorBufferInfo::default().buffer(buf_bias.buffer).offset(buf_bias.pool_offset.unwrap_or(0)).range(buf_bias.size)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).range(buf_out.size)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_s),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_bias),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        let use_fast = dtype == DataType::BitNet2 || dtype == DataType::BitNet1_6;
        let pipe = if use_fast { backend.pipe_bit_linear_lut } else { backend.pipe_bit_linear };

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_bit_linear, 0, &[set], &[]);

        let has_bias = if bias_raw.is_empty() { 0u32 } else { 1u32 };
        let pc_data = if use_fast {
            let dt = match dtype {
                DataType::BitNet2 => 100u32,
                DataType::BitNet1_6 => 101u32,
                _ => 0u32,
            };
            vec![m, k, n, has_bias, dt]
        } else {
            vec![m, k, n, has_bias]
        };
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_bit_linear, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc_data));

        backend.device.cmd_dispatch(cmd, (n + 15) / 16, (m + 15) / 16, 1);

        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).size(buf_out.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: buf_out.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_out }]);

        backend.device.end_command_buffer(cmd).unwrap();
        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(std::slice::from_ref(&wait_val));
        let submit_info = vk::SubmitInfo::default().push_next(&mut timeline_info).command_buffers(std::slice::from_ref(&cmd)).signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));
        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();

        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_a, stage_b, stage_s, stage_bias, stage_out.copy_for_async()],
            device_buffers: vec![buf_a, buf_b, buf_s, buf_bias, buf_out],
            cmd_buffer: cmd,
            wait_id: wait_val,
        });

        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(std::slice::from_ref(&backend.timeline_semaphore)).values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        download_from_stage_raw(out_raw, &stage_out);
    }
    poll_async_ops();
}
