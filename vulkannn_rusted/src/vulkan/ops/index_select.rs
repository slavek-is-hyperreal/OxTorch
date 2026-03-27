use ash::vk;
use std::sync::atomic::Ordering;
use crate::tensor::DataType;
use crate::vulkan::context::{BACKEND, AsyncOp, poll_async_ops};
use crate::vulkan::memory::{get_buffer, get_buffer_readback};
use crate::vulkan::ops::{begin_cmd, upload_to_stage, upload_to_stage_raw, download_from_stage};

pub fn execute_index_select_into(weight_raw: &[u8], indices_raw: &[u8], out_raw: &mut [u8], num_indices: u32, feature_len: u32, dtype: DataType) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let bytes_per_elem = match dtype {
        DataType::F32  => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    
    let num_weight_elements = weight_raw.len() / bytes_per_elem;
    let num_weight_bytes_f32 = (num_weight_elements * 4) as vk::DeviceSize;
    
    let num_out_elements = (num_indices * feature_len) as usize;
    let num_out_bytes_f32 = (num_out_elements * 4) as vk::DeviceSize;
    
    let num_indices_bytes = (num_indices * 4) as vk::DeviceSize;

    let buf_weight = get_buffer(num_weight_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("IdxSel_Weight"), false);
    let buf_indices = get_buffer(num_indices_bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("IdxSel_Indices"), false);
    let buf_out = get_buffer(num_out_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("IdxSel_Out"), false);

    let stage_weight = get_buffer(num_weight_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("IdxSel_Stage_W"), true);
    let stage_indices = get_buffer(num_indices_bytes, vk::BufferUsageFlags::TRANSFER_SRC, Some("IdxSel_Stage_I"), true);
    let stage_out = get_buffer_readback(num_out_bytes_f32, vk::BufferUsageFlags::TRANSFER_DST, Some("IdxSel_Stage_Out"));

    upload_to_stage(weight_raw, &stage_weight, dtype);
    upload_to_stage_raw(indices_raw, &stage_indices);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_weight.buffer, buf_weight.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_weight.pool_offset.unwrap_or(0), size: num_weight_bytes_f32 }]);
        backend.device.cmd_copy_buffer(cmd, stage_indices.buffer, buf_indices.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_indices.pool_offset.unwrap_or(0), size: num_indices_bytes }]);
        
        let barriers = [
            vk::BufferMemoryBarrier::default().buffer(buf_weight.buffer).offset(buf_weight.pool_offset.unwrap_or(0)).size(buf_weight.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            vk::BufferMemoryBarrier::default().buffer(buf_indices.buffer).offset(buf_indices.pool_offset.unwrap_or(0)).size(buf_indices.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
        ];
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

        let set = backend.pool_desc_index_select.lock().unwrap().next();
        let info_w = [vk::DescriptorBufferInfo::default().buffer(buf_weight.buffer).offset(buf_weight.pool_offset.unwrap_or(0)).range(buf_weight.size)];
        let info_i = [vk::DescriptorBufferInfo::default().buffer(buf_indices.buffer).offset(buf_indices.pool_offset.unwrap_or(0)).range(buf_indices.size)];
        let info_o = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).range(buf_out.size)];
        
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_w),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_i),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_o),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        let pc_data = [num_indices, feature_len];
        let pc_bytes = bytemuck::cast_slice(&pc_data);
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_index_select, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_index_select);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_index_select, 0, &[set], &[]);
        let workgroups = ((num_indices * feature_len) + 255) / 256;
        backend.device.cmd_dispatch(cmd, workgroups, 1, 1);
        
        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).size(buf_out.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        
        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: buf_out.pool_offset.unwrap_or(0), dst_offset: 0, size: num_out_bytes_f32 }]);
        
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
            staging_buffers: vec![stage_weight, stage_indices, stage_out.copy_for_async()],
            device_buffers: vec![buf_weight, buf_indices, buf_out],
            cmd_buffer: cmd,
            wait_id: wait_val,
        });

        let target = wait_val;
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .values(std::slice::from_ref(&target));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
        
        download_from_stage(out_raw, &stage_out, dtype); 
    }
    
    poll_async_ops();
}
