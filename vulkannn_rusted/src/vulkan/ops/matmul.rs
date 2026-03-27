use ash::vk;
use std::sync::atomic::Ordering;
use crate::tensor::DataType;
use crate::vulkan::context::{BACKEND, AsyncOp, poll_async_ops};
use crate::vulkan::memory::{get_buffer, get_buffer_readback};
use crate::vulkan::ops::{begin_cmd, upload_to_stage, download_from_stage};

pub fn execute_linear_into(a_raw: &[u8], b_raw: &[u8], bias_raw: &[u8], res_raw: &mut [u8], m: u32, k: u32, n: u32, act_type: u32, transpose_b: u32, dtype: DataType) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let num_bytes_f32_a = (m * k * 4) as vk::DeviceSize;
    let num_bytes_f32_b = (k * n * 4) as vk::DeviceSize;
    let num_bytes_f32_c = (m * n * 4) as vk::DeviceSize;
    let num_bytes_f32_bias = (n * 4) as vk::DeviceSize;

    let buf_a = get_buffer(num_bytes_f32_a, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_A"), false);
    let buf_b = get_buffer(num_bytes_f32_b, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_B"), false);
    let buf_bias = get_buffer(num_bytes_f32_bias, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_Bias"), false);
    let buf_c = get_buffer(num_bytes_f32_c, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_C"), false);

    let stage_a = get_buffer(num_bytes_f32_a, vk::BufferUsageFlags::TRANSFER_SRC, Some("Linear_Stage_A"), true);
    let stage_b = get_buffer(num_bytes_f32_b, vk::BufferUsageFlags::TRANSFER_SRC, Some("Linear_Stage_B"), true);
    let stage_bias = get_buffer(num_bytes_f32_bias, vk::BufferUsageFlags::TRANSFER_SRC, Some("Linear_Stage_Bias"), true);
    let stage_c = get_buffer_readback(num_bytes_f32_c, vk::BufferUsageFlags::TRANSFER_DST, Some("Linear_Stage_C"));

    upload_to_stage(a_raw, &stage_a, dtype);
    upload_to_stage(b_raw, &stage_b, dtype);
    if bias_raw.is_empty() {
        let ptr = stage_bias.mapped_ptr.unwrap() as *mut f32;
        let dst_slice = unsafe { std::slice::from_raw_parts_mut(ptr, n as usize) };
        dst_slice.fill(0.0);
    } else {
        upload_to_stage(bias_raw, &stage_bias, dtype);
    }

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_a.pool_offset.unwrap_or(0), size: num_bytes_f32_a }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_b.pool_offset.unwrap_or(0), size: num_bytes_f32_b }]);
        backend.device.cmd_copy_buffer(cmd, stage_bias.buffer, buf_bias.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_bias.pool_offset.unwrap_or(0), size: num_bytes_f32_bias }]);

        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).size(buf_a.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).size(buf_b.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_bias.buffer).offset(buf_bias.pool_offset.unwrap_or(0)).size(buf_bias.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_linear.lock().unwrap().next();
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).range(buf_a.size)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).range(buf_b.size)];
        let info_c = [vk::DescriptorBufferInfo::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).range(buf_c.size)];
        let info_bias = [vk::DescriptorBufferInfo::default().buffer(buf_bias.buffer).offset(buf_bias.pool_offset.unwrap_or(0)).range(buf_bias.size)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_c),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_bias),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_matmul);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_matmul, 0, &[set], &[]);

        let pc_data = [m, k, n, act_type, 1, transpose_b]; // has_bias = 1
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_matmul, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc_data));

        backend.device.cmd_dispatch(cmd, (n + 15) / 16, (m + 15) / 16, 1);

        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).size(buf_c.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_c.buffer, stage_c.buffer, &[vk::BufferCopy { src_offset: buf_c.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32_c }]);

        backend.device.end_command_buffer(cmd).unwrap();
        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(std::slice::from_ref(&wait_val));
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_info)
            .command_buffers(&cmds)
            .signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));

        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();

        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_a, stage_b, stage_bias, stage_c.copy_for_async()],
            device_buffers: vec![buf_a, buf_b, buf_bias, buf_c],
            cmd_buffer: cmd,
            wait_id: wait_val,
        });

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        download_from_stage(res_raw, &stage_c, dtype);
    }

    poll_async_ops();
}

pub fn execute_matmul_into(a_raw: &[u8], b_raw: &[u8], res_raw: &mut [u8], batch: u32, m: u32, k: u32, n: u32, dtype: DataType) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let num_bytes_f32_a = (batch * m * k * 4) as vk::DeviceSize;
    let num_bytes_f32_b = (batch * k * n * 4) as vk::DeviceSize;
    let num_bytes_f32_c = (batch * m * n * 4) as vk::DeviceSize;

    let buf_a = get_buffer(num_bytes_f32_a, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_A"), false);
    let buf_b = get_buffer(num_bytes_f32_b, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_B"), false);
    let buf_c = get_buffer(num_bytes_f32_c, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_C"), false);

    let stage_a = get_buffer(num_bytes_f32_a, vk::BufferUsageFlags::TRANSFER_SRC, Some("MatMul_Stage_A"), true);
    let stage_b = get_buffer(num_bytes_f32_b, vk::BufferUsageFlags::TRANSFER_SRC, Some("MatMul_Stage_B"), true);
    let stage_c = get_buffer_readback(num_bytes_f32_c, vk::BufferUsageFlags::TRANSFER_DST, Some("MatMul_Stage_C"));

    upload_to_stage(a_raw, &stage_a, dtype);
    upload_to_stage(b_raw, &stage_b, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_a.pool_offset.unwrap_or(0), size: num_bytes_f32_a }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_b.pool_offset.unwrap_or(0), size: num_bytes_f32_b }]);

        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).size(buf_a.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).size(buf_b.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_matmul.lock().unwrap().next();
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).range(buf_a.size)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).range(buf_b.size)];
        let info_c = [vk::DescriptorBufferInfo::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).range(buf_c.size)];
        
        // Use a dummy empty buffer for bias if not needed? 
        // Or re-use buf_c offset 0?
        // Shader expects 4 bindings.
        let info_bias = [vk::DescriptorBufferInfo::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).range(4)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_c),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_bias),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_matmul);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_matmul, 0, &[set], &[]);

        let pc_data = [m, k, n, 0, 0, 0]; // act_type = 0, has_bias = 0, transpose_b = 0
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_matmul, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc_data));

        backend.device.cmd_dispatch(cmd, (n + 15) / 16, (m + 15) / 16, 1);

        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).size(buf_c.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        
        backend.device.cmd_copy_buffer(cmd, buf_c.buffer, stage_c.buffer, &[vk::BufferCopy { src_offset: buf_c.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32_c }]);
        
        backend.device.end_command_buffer(cmd).unwrap();

        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(std::slice::from_ref(&wait_val));

        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_info)
            .command_buffers(&cmds)
            .signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));
            
        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();
        
        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_a, stage_b, stage_c.copy_for_async()],
            device_buffers: vec![buf_a, buf_b, buf_c],
            cmd_buffer: cmd,
            wait_id: wait_val,
        });

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        
        download_from_stage(res_raw, &stage_c, dtype); 
    }
    
    poll_async_ops();
}
