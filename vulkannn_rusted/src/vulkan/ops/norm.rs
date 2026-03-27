use ash::vk;
use std::sync::atomic::Ordering;
use crate::tensor::DataType;
use crate::vulkan::context::{BACKEND, AsyncOp, poll_async_ops, poll_async_ops_until};
use crate::vulkan::memory::{get_buffer, get_buffer_readback};
use crate::vulkan::ops::{begin_cmd, upload_to_stage, download_from_stage};

pub fn execute_layer_norm_into(x_raw: &[u8], w_raw: &[u8], b_raw: &[u8], out_raw: &mut [u8], n: u32, d: u32, eps: f32, dtype: DataType) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let num_bytes_f32_x = (n * d * 4) as vk::DeviceSize;
    let num_bytes_f32_w = (d * 4) as vk::DeviceSize;

    let buf_x = get_buffer(num_bytes_f32_x, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("LayerNorm_X"), false);
    let buf_w = get_buffer(num_bytes_f32_w, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("LayerNorm_W"), false);
    let buf_b = get_buffer(num_bytes_f32_w, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("LayerNorm_B"), false);
    let buf_y = get_buffer(num_bytes_f32_x, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("LayerNorm_Y"), false);

    let stage_x = get_buffer(num_bytes_f32_x, vk::BufferUsageFlags::TRANSFER_SRC, Some("LayerNorm_Stage_X"), true);
    let stage_w = get_buffer(num_bytes_f32_w, vk::BufferUsageFlags::TRANSFER_SRC, Some("LayerNorm_Stage_W"), true);
    let stage_b = get_buffer(num_bytes_f32_w, vk::BufferUsageFlags::TRANSFER_SRC, Some("LayerNorm_Stage_B"), true);
    let stage_y = get_buffer_readback(num_bytes_f32_x, vk::BufferUsageFlags::TRANSFER_DST, Some("LayerNorm_Stage_Y"));

    upload_to_stage(x_raw, &stage_x, dtype);
    
    if !w_raw.is_empty() { upload_to_stage(w_raw, &stage_w, dtype); }
    else { let ptr = stage_w.mapped_ptr.unwrap() as *mut f32; unsafe { for i in 0..d as usize { ptr.add(i).write(1.0); } } }
    
    if !b_raw.is_empty() { upload_to_stage(b_raw, &stage_b, dtype); } 
    else { let ptr = stage_b.mapped_ptr.unwrap() as *mut f32; unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, num_bytes_f32_w as usize); } }

    unsafe {
        let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
        backend.device.cmd_copy_buffer(cmd, stage_x.buffer, buf_x.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_x.pool_offset.unwrap_or(0), size: num_bytes_f32_x }]);
        backend.device.cmd_copy_buffer(cmd, stage_w.buffer, buf_w.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_w.pool_offset.unwrap_or(0), size: num_bytes_f32_w }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_b.pool_offset.unwrap_or(0), size: num_bytes_f32_w }]);

        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_x.buffer).offset(buf_x.pool_offset.unwrap_or(0)).size(buf_x.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_w.buffer).offset(buf_w.pool_offset.unwrap_or(0)).size(buf_w.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).size(buf_b.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_layer_norm.lock().unwrap().next();
        let info_x = [vk::DescriptorBufferInfo::default().buffer(buf_x.buffer).offset(buf_x.pool_offset.unwrap_or(0)).range(buf_x.size)];
        let info_w = [vk::DescriptorBufferInfo::default().buffer(buf_w.buffer).offset(buf_w.pool_offset.unwrap_or(0)).range(buf_w.size)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).range(buf_b.size)];
        let info_y = [vk::DescriptorBufferInfo::default().buffer(buf_y.buffer).offset(buf_y.pool_offset.unwrap_or(0)).range(buf_y.size)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_x),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_w),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_y),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layer_norm);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_layer_norm, 0, &[set], &[]);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct PushConsts { n: u32, d: u32, eps: f32 }
        unsafe impl bytemuck::Zeroable for PushConsts {}
        unsafe impl bytemuck::Pod for PushConsts {}
        let pc_data = PushConsts { n, d, eps };
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_layer_norm, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&[pc_data]));

        backend.device.cmd_dispatch(cmd, n, 1, 1);

        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_y.buffer).offset(buf_y.pool_offset.unwrap_or(0)).size(buf_y.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        backend.device.cmd_copy_buffer(cmd, buf_y.buffer, stage_y.buffer, &[vk::BufferCopy { src_offset: buf_y.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32_x }]);

        backend.device.end_command_buffer(cmd).unwrap();
        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(std::slice::from_ref(&wait_val));
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().push_next(&mut timeline_info).command_buffers(&cmds).signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));

        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();

        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_x, stage_w, stage_b, stage_y.copy_for_async()],
            device_buffers: vec![buf_x, buf_w, buf_b, buf_y],
            cmd_buffer: cmd,
            wait_id: wait_val,
        });

        poll_async_ops_until(wait_val);
        download_from_stage(out_raw, &stage_y, dtype);
    }
    poll_async_ops();
}

pub fn execute_rms_norm_into(x_raw: &[u8], w_raw: &[u8], out_raw: &mut [u8], n: u32, d: u32, eps: f32, dtype: DataType) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let num_bytes_f32_x = (n * d * 4) as vk::DeviceSize;
    let num_bytes_f32_w = (d * 4) as vk::DeviceSize;

    let buf_x = get_buffer(num_bytes_f32_x, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("RMSNorm_X"), false);
    let buf_w = get_buffer(num_bytes_f32_w, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("RMSNorm_W"), false);
    let buf_y = get_buffer(num_bytes_f32_x, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("RMSNorm_Y"), false);

    let stage_x = get_buffer(num_bytes_f32_x, vk::BufferUsageFlags::TRANSFER_SRC, Some("RMSNorm_Stage_X"), true);
    let stage_w = get_buffer(num_bytes_f32_w, vk::BufferUsageFlags::TRANSFER_SRC, Some("RMSNorm_Stage_W"), true);
    let stage_y = get_buffer_readback(num_bytes_f32_x, vk::BufferUsageFlags::TRANSFER_DST, Some("RMSNorm_Stage_Y"));

    upload_to_stage(x_raw, &stage_x, dtype);
    
    if !w_raw.is_empty() { upload_to_stage(w_raw, &stage_w, dtype); }
    else { let ptr = stage_w.mapped_ptr.unwrap() as *mut f32; unsafe { for i in 0..d as usize { ptr.add(i).write(1.0); } } }

    unsafe {
        let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
        backend.device.cmd_copy_buffer(cmd, stage_x.buffer, buf_x.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_x.pool_offset.unwrap_or(0), size: num_bytes_f32_x }]);
        backend.device.cmd_copy_buffer(cmd, stage_w.buffer, buf_w.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_w.pool_offset.unwrap_or(0), size: num_bytes_f32_w }]);

        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_x.buffer).offset(buf_x.pool_offset.unwrap_or(0)).size(buf_x.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_w.buffer).offset(buf_w.pool_offset.unwrap_or(0)).size(buf_w.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_rms_norm.lock().unwrap().next();
        let info_x = [vk::DescriptorBufferInfo::default().buffer(buf_x.buffer).offset(buf_x.pool_offset.unwrap_or(0)).range(buf_x.size)];
        let info_w = [vk::DescriptorBufferInfo::default().buffer(buf_w.buffer).offset(buf_w.pool_offset.unwrap_or(0)).range(buf_w.size)];
        let info_y = [vk::DescriptorBufferInfo::default().buffer(buf_y.buffer).offset(buf_y.pool_offset.unwrap_or(0)).range(buf_y.size)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_x),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_w),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_y),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_rms_norm);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_rms_norm, 0, &[set], &[]);

        #[repr(C)]
        #[derive(Copy, Clone)]
        struct PushConsts { n: u32, d: u32, eps: f32 }
        unsafe impl bytemuck::Zeroable for PushConsts {}
        unsafe impl bytemuck::Pod for PushConsts {}
        let pc_data = PushConsts { n, d, eps };
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_rms_norm, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&[pc_data]));

        backend.device.cmd_dispatch(cmd, n, 1, 1);

        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_y.buffer).offset(buf_y.pool_offset.unwrap_or(0)).size(buf_y.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        backend.device.cmd_copy_buffer(cmd, buf_y.buffer, stage_y.buffer, &[vk::BufferCopy { src_offset: buf_y.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32_x }]);

        backend.device.end_command_buffer(cmd).unwrap();
        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(std::slice::from_ref(&wait_val));
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().push_next(&mut timeline_info).command_buffers(&cmds).signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore));

        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();

        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_x, stage_w, stage_y.copy_for_async()],
            device_buffers: vec![buf_x, buf_w, buf_y],
            cmd_buffer: cmd,
            wait_id: wait_val,
        });

        poll_async_ops_until(wait_val);
        download_from_stage(out_raw, &stage_y, dtype);
    }
    poll_async_ops();
}
