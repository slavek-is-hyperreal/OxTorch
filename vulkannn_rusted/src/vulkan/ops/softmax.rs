use ash::vk;
use std::sync::atomic::Ordering;
use crate::tensor::DataType;
use crate::vulkan::context::{BACKEND, AsyncOp, poll_async_ops};
use crate::vulkan::memory::{get_buffer, get_buffer_readback};
use crate::vulkan::ops::{begin_cmd, upload_to_stage, download_from_stage};

pub fn execute_softmax_into(input_raw: &[u8], output_raw: &mut [u8], width: u32, height: u32, is_log: bool, dtype: DataType) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let num_bytes_f32 = (width * height * 4) as vk::DeviceSize;

    let buf_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Softmax_In"), false);
    // buf_out must not alias buf_in in the pool — force it to a fresh standalone buffer
    let buf_out_size = num_bytes_f32 + 1; 
    let buf_out = get_buffer(buf_out_size, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Softmax_Out"), false);
    let stage_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Softmax_Stage_In"), true);
    let stage_out = get_buffer_readback(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_DST, Some("Softmax_Stage_Out"));

    upload_to_stage(input_raw, &stage_in, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_in.buffer, buf_in.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_in.pool_offset.unwrap_or(0), size: num_bytes_f32 }]);

        let barrier_in = vk::BufferMemoryBarrier::default()
            .buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).size(buf_in.size)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier_in], &[]);

        let set = backend.pool_desc_act.lock().unwrap().next();
        let info_in  = [vk::DescriptorBufferInfo::default().buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).range(buf_in.size)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).range(buf_out.size)];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_in),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_softmax);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_act, 0, &[set], &[]);

        let pc_data: [u32; 4] = [width, height, if is_log { 1 } else { 0 }, 0];
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_act, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc_data));

        backend.device.cmd_dispatch(cmd, height, 1, 1);

        let barrier_out = vk::BufferMemoryBarrier::default()
            .buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).size(buf_out.size)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: buf_out.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32 }]);
        
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
            staging_buffers: vec![stage_in, stage_out.copy_for_async()],
            device_buffers: vec![buf_in, buf_out],
            cmd_buffer: cmd,
            wait_id: wait_val,
        });

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        
        download_from_stage(output_raw, &stage_out, dtype); 
    }
    
    poll_async_ops();
}
