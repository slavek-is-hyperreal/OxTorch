use ash::vk;
use std::sync::atomic::Ordering;
use crate::tensor::DataType;
use crate::vulkan::context::{BACKEND, AsyncOp, poll_async_ops};
use crate::vulkan::memory::{get_buffer, get_buffer_readback};
use crate::vulkan::ops::{begin_cmd, upload_to_stage, download_from_stage};

pub fn execute_elementwise_into(a_raw: &[u8], b_raw: &[u8], res_raw: &mut [u8], op_id: u32, dtype: DataType) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let bytes_per_elem = match dtype {
        DataType::F32  => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let num_elements = a_raw.len() / bytes_per_elem;
    let num_bytes_f32 = (num_elements * 4) as vk::DeviceSize;

    let buf_a = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Elem_A"), false);
    let buf_b = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Elem_B"), false);
    let buf_c = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Elem_C"), false);

    let stage_a = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Elem_Stage_A"), true);
    let stage_b = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Elem_Stage_B"), true);
    let stage_c = get_buffer_readback(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_DST, Some("Elem_Stage_C"));

    upload_to_stage(a_raw, &stage_a, dtype);
    upload_to_stage(b_raw, &stage_b, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_a.buffer, buf_a.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_a.pool_offset.unwrap_or(0), size: num_bytes_f32 }]);
        backend.device.cmd_copy_buffer(cmd, stage_b.buffer, buf_b.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_b.pool_offset.unwrap_or(0), size: num_bytes_f32 }]);
        
        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).size(buf_a.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).size(buf_b.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_elementwise.lock().unwrap().next();
        let info_a = [vk::DescriptorBufferInfo::default().buffer(buf_a.buffer).offset(buf_a.pool_offset.unwrap_or(0)).range(buf_a.size)];
        let info_b = [vk::DescriptorBufferInfo::default().buffer(buf_b.buffer).offset(buf_b.pool_offset.unwrap_or(0)).range(buf_b.size)];
        let info_c = [vk::DescriptorBufferInfo::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).range(buf_c.size)];
        
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_a),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_b),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_c),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        let mut pc = [0u32; 4];
        pc[0] = num_elements as u32;
        pc[1] = op_id;
        let pc_bytes = bytemuck::cast_slice(&pc);
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_elementwise, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_elementwise);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_elementwise, 0, &[set], &[]);
        let workgroups = (num_elements as u32 + 255) / 256;
        backend.device.cmd_dispatch(cmd, workgroups, 1, 1);
        
        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_c.buffer).offset(buf_c.pool_offset.unwrap_or(0)).size(buf_c.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        
        backend.device.cmd_copy_buffer(cmd, buf_c.buffer, stage_c.buffer, &[vk::BufferCopy { src_offset: buf_c.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_f32 }]);
        
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

        let target = wait_val;
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&target));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        
        download_from_stage(res_raw, &stage_c, dtype); 
    }
    
    poll_async_ops();
}
