use ash::vk;
use std::sync::atomic::Ordering;
use crate::tensor::DataType;
use crate::vulkan::context::{BACKEND, AsyncOp, poll_async_ops_until};
use crate::vulkan::memory::{get_buffer, get_buffer_readback, CachedBuffer};
use crate::vulkan::ops::{begin_cmd, upload_to_stage, download_from_stage};

pub fn execute_activation_into(input_raw: &[u8], op: &str, param1: f32, param2: f32, res_raw: &mut [u8], dtype: DataType, _is_hybrid: bool, _use_staging: bool) {
    let t_start = std::time::Instant::now();
    let (wait_id, stage_out) = submit_activation_into(input_raw, op, param1, param2, res_raw, dtype, _is_hybrid, _use_staging);
    poll_async_ops_until(wait_id);

    download_from_stage(res_raw, &stage_out, dtype);

    static PRINT_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !PRINT_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        println!("\n[VNN PERF] Act Sync Call Total Block Time: {:.2}ms", t_start.elapsed().as_secs_f64()*1000.0);
    }
}

pub fn submit_activation_into(input_raw: &[u8], op: &str, param1: f32, param2: f32, _res_raw: &mut [u8], dtype: DataType, _is_hybrid: bool, _use_staging: bool) -> (u64, CachedBuffer) {
    let backend = BACKEND.get().expect("Vulkan Backend not initialized");
    let num_bytes = input_raw.len() as vk::DeviceSize;
    let bytes_per_elem = match dtype {
        DataType::F32 => 4,
        DataType::Int8 => 1,
        _ => 2,
    };
    let num_elements = num_bytes as u32 / bytes_per_elem;
    let num_bytes_staging = (num_elements * 4) as vk::DeviceSize;
    
    let buf_in = get_buffer(num_bytes_staging, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Some("Act_In"), false);
    let buf_out = get_buffer(num_bytes_staging, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, Some("Act_Out"), false);

    let stage_in = get_buffer(num_bytes_staging, vk::BufferUsageFlags::TRANSFER_SRC, Some("Act_Stage_In"), true);
    let stage_out = get_buffer_readback(num_bytes_staging, vk::BufferUsageFlags::TRANSFER_DST, Some("Act_Stage_Out"));

    upload_to_stage(input_raw, &stage_in, dtype);

    let cmd = begin_cmd(&backend.device, backend.compute_cmd_pool);
    unsafe {
        backend.device.cmd_copy_buffer(cmd, stage_in.buffer, buf_in.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: buf_in.pool_offset.unwrap_or(0), size: num_bytes_staging }]);
        
        let barrier = vk::BufferMemoryBarrier::default().buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).size(buf_in.size).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier], &[]);

        let set = backend.pool_desc_act.lock().unwrap().next();
        
        let info_in = [vk::DescriptorBufferInfo::default().buffer(buf_in.buffer).offset(buf_in.pool_offset.unwrap_or(0)).range(buf_in.size)];
        let info_out = [vk::DescriptorBufferInfo::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).range(buf_out.size)];
        
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_in),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info_out),
        ];
        backend.device.update_descriptor_sets(&writes, &[]);

        let pipe = match op {
            "relu" => backend.pipe_relu,
            "sigmoid" => backend.pipe_sigmoid,
            "silu" => backend.pipe_silu,
            "gelu" => backend.pipe_gelu,
            "leaky_relu" => backend.pipe_leaky_relu,
            "elu" => backend.pipe_elu,
            "tanh" => backend.pipe_tanh,
            "clamp" => backend.pipe_clamp,
            "neg" => backend.pipe_neg,
            "pow" => backend.pipe_pow,
            _ => panic!("Unsupported activation OP: {}", op),
        };

        backend.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        backend.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, backend.pipe_layout_act, 0, &[set], &[]);
        
        let pc_data = [num_elements as f32, 0.0, param1, param2];
        let pc_bytes = bytemuck::cast_slice(&pc_data);
        backend.device.cmd_push_constants(cmd, backend.pipe_layout_act, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        
        backend.device.cmd_dispatch(cmd, (num_elements + 255) / 256, 1, 1);
        
        let barrier_out = vk::BufferMemoryBarrier::default().buffer(buf_out.buffer).offset(buf_out.pool_offset.unwrap_or(0)).size(buf_out.size).src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        backend.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[barrier_out], &[]);
        
        backend.device.cmd_copy_buffer(cmd, buf_out.buffer, stage_out.buffer, &[vk::BufferCopy { src_offset: buf_out.pool_offset.unwrap_or(0), dst_offset: 0, size: num_bytes_staging }]);
        
        backend.device.end_command_buffer(cmd).unwrap();
        
        let wait_id = backend.timeline_value.fetch_add(1, Ordering::SeqCst) + 1;
        
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(std::slice::from_ref(&wait_id));
            
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .signal_semaphores(std::slice::from_ref(&backend.timeline_semaphore))
            .push_next(&mut timeline_info);
            
        backend.device.queue_submit(backend.compute_queue, &[submit_info], vk::Fence::null()).unwrap();
        
        let async_stage_out = stage_out.copy_for_async();

        backend.pending_ops.lock().unwrap().push(AsyncOp {
            staging_buffers: vec![stage_in, async_stage_out],
            device_buffers: vec![buf_in, buf_out],
            cmd_buffer: cmd,
            wait_id,
        });
        
        (wait_id, stage_out.copy_for_async())
    }
}
