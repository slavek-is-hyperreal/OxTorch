use ash::vk;
use std::sync::atomic::Ordering;
use crate::tensor::DataType;
use crate::vulkan::context::{BACKEND, AsyncOp, poll_async_ops};
use crate::vulkan::memory::{get_buffer, get_buffer_readback};
use crate::vulkan::ops::{begin_cmd, upload_to_stage, download_from_stage};

pub fn execute_reduce(input_raw: &[u8], op: &str, dtype: DataType) -> Vec<f32> {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let bytes_per_elem = match dtype {
        DataType::F32 => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let elem_count = input_raw.len() / bytes_per_elem;
    let num_bytes_f32 = (elem_count * 4) as vk::DeviceSize;
    let num_blocks = (elem_count + 255) / 256;
    let out_elements = if op == "argmax" { num_blocks * 2 } else { num_blocks };
    let out_num_bytes = (out_elements * 4) as vk::DeviceSize;

    let buf_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Reduce_In"),  false);
    let buf_out = get_buffer(out_num_bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Reduce_Out"), false);
    let stage_in  = get_buffer(num_bytes_f32, vk::BufferUsageFlags::TRANSFER_SRC, Some("Reduce_Stage_In"),  true);
    let stage_out = get_buffer_readback(out_num_bytes, vk::BufferUsageFlags::TRANSFER_DST, Some("Reduce_Stage_Out"));

    upload_to_stage(input_raw, &stage_in, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_in.buffer, buf_in.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_in.pool_offset.unwrap_or(0), size: num_bytes_f32 }]);

        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).size(buf_in.size)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier], &[]);

        let set = backend.pool_desc_reduce.lock().unwrap().next();

        let info_in  = [vk::DescriptorBufferInfo::default().buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).range(buf_in.size)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).range(buf_out.size)];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_in),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        let pipe = match op {
            "sum" | "mean" => backend.pipe_reduce_sum, 
            "max" => backend.pipe_reduce_max,
            "min" => backend.pipe_reduce_min,
            "argmax" => backend.pipe_reduce_argmax,
            _ => panic!("Unsupported reduction OP: {}", op),
        };

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_reduce, 0, &[set], &[]);
        
        let pc_data = [elem_count as u32, 0u32, 0u32]; 
        let pc_bytes = bytemuck::cast_slice(&pc_data);
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_reduce, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        
        backend.device.cmd_dispatch(cmd, num_blocks as u32, 1, 1);

        let barrier_out = vk::BufferMemoryBarrier::default()
            .buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).size(buf_out.size)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);

        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: buf_out.pool_offset.unwrap_or(0), dst_offset: 0, size: out_num_bytes }]);

        backend.device.end_command_buffer(cmd).unwrap();

        let wait_val = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(std::slice::from_ref(&wait_val));
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

        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(std::slice::from_ref(&backend.timeline_semaphore)).values(std::slice::from_ref(&wait_val));
        backend.device.wait_semaphores(&wait_info, u64::MAX).unwrap();
    }

    let mut out_bytes = vec![0u8; out_num_bytes as usize];
    download_from_stage(&mut out_bytes, &stage_out, DataType::F32);
    poll_async_ops();

    let partials: &[f32] = bytemuck::cast_slice(&out_bytes);
    if op == "argmax" {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0.0f32;
        for i in 0..(out_bytes.len() / 8) {
            let val = partials[i * 2];
            let idx = partials[i * 2 + 1];
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        vec![max_idx]
    } else {
        partials.to_vec()
    }
}
