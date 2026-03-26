use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::cmp::min;
use crate::tensor::{Tensor, DataType};

#[derive(Debug, Serialize, Deserialize)]
pub struct BitNetConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
}

pub struct BitNetRMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl BitNetRMSNorm {
    pub fn forward(&self, x: &Tensor) -> PyResult<Tensor> {
        x.py_rms_norm(vec![x.shape[x.shape.len()-1]], Some(&self.weight), self.eps)
    }
}

pub struct BitLinear {
    pub weight: Tensor,
    pub scale: Tensor,
    pub bias: Option<Tensor>,
}

impl BitLinear {
    pub fn forward(&self, x: &Tensor) -> PyResult<Tensor> {
        let (x_q, s_a) = x.execute_quantize_per_token()?;
        let mut y = Tensor::execute_bit_linear_with_n(&x_q, &self.weight, self.weight.shape[0], &self.scale, self.bias.as_ref())?;
        y = y.execute_mul_broadcast(&s_a)?;
        // println!("[BitLinear] {:?} FWD Final Shape: {:?}", self.weight.name, y.shape);
        Ok(y)
    }
}

pub struct BitNetLayer {
    pub q_proj: BitLinear,
    pub k_proj: BitLinear,
    pub v_proj: BitLinear,
    pub o_proj: BitLinear,
    pub gate_proj: BitLinear,
    pub up_proj: BitLinear,
    pub down_proj: BitLinear,
    pub input_layernorm: BitNetRMSNorm,
    pub post_attention_layernorm: BitNetRMSNorm,
    pub attn_sub_norm: BitNetRMSNorm,
    pub ffn_sub_norm: BitNetRMSNorm,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl BitNetLayer {
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, past_k: Option<&Tensor>, past_v: Option<&Tensor>) -> PyResult<(Tensor, Tensor, Tensor)> {
        let residual = x.clone();
        let x_norm = self.input_layernorm.forward(x)?;
        
        let seq_len = x_norm.shape[1];
        // println!("[BitNet] Layer FWD heads={}, kv_heads={}, head_dim={}, seq={}", self.n_heads, self.n_kv_heads, self.head_dim, seq_len);
        
        let q_res = self.q_proj.forward(&x_norm)?;
        let q = q_res.execute_reshape(vec![1, seq_len, self.n_heads, self.head_dim])?;
        
        let k_res = self.k_proj.forward(&x_norm)?;
        println!("[BitNet] K_PROJ Output Size: {}", k_res.shape.iter().product::<usize>());
        let k = k_res.execute_reshape(vec![1, seq_len, self.n_kv_heads, self.head_dim])?;
        
        let v_res = self.v_proj.forward(&x_norm)?;
        let v = v_res.execute_reshape(vec![1, seq_len, self.n_kv_heads, self.head_dim])?;
        
        println!("[BitNet] FWD RoPE application...");
        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;

        let (k, v) = match (past_k, past_v) {
            (Some(pk), Some(pv)) => {
                let new_k = Tensor::execute_cat(&[pk, &k], 1)?;
                let new_v = Tensor::execute_cat(&[pv, &v], 1)?;
                (new_k, new_v)
            },
            _ => (k, v)
        };
        
        let (k_rep, v_rep) = if self.n_kv_heads < self.n_heads {
            let kr = k.execute_repeat_interleave(self.n_heads / self.n_kv_heads, 2)?;
            let vr = v.execute_repeat_interleave(self.n_heads / self.n_kv_heads, 2)?;
            (kr, vr)
        } else {
            (k.clone(), v.clone())
        };
        
        // println!("[BitNet] FWD Scaled Dot Product attention sequence...");
        // SDPA: transpose to (bs=1, n_heads, seq, head_dim)
        let q_tp = q.execute_transpose_dims(1, 2)?;
        let k_tp = k_rep.execute_transpose_dims(1, 2)?;
        let v_tp = v_rep.execute_transpose_dims(1, 2)?;
        
        let mut attn_out = self.scaled_dot_product_attention(&q_tp, &k_tp, &v_tp)?;
        
        // Transpose back: (1, n_heads, seq, head_dim) → (1, seq, n_heads, head_dim)
        attn_out = attn_out.execute_transpose_dims(1, 2)?;
        // Flatten to [seq_len, n_heads*head_dim]
        let attn_out_flat = attn_out.execute_reshape(vec![1, seq_len, self.n_heads * self.head_dim])?;
        
        // SubLN + Output Projection
        let attn_norm = self.attn_sub_norm.forward(&attn_out_flat)?;
        let o_out = self.o_proj.forward(&attn_norm)?;
        // println!("[BitNet] Layer Attn o_proj shape: {:?}", o_out.shape);
        let mut x = residual.elementwise_op(&o_out, "add")?;
        
        /*
        if x.shape.len() > 1 && x.shape[1] == 1 {
            println!("[BitNet] WARNING! Shape collapse detected at end of Attn residuals: {:?}", x.shape);
        }
        */
        
        // 2. MLP
        let residual_mlp = x.clone();
        let x_norm_mlp = self.post_attention_layernorm.forward(&x)?;
        
        let gate = self.gate_proj.forward(&x_norm_mlp)?;
        let up = self.up_proj.forward(&x_norm_mlp)?;
        
        // ReLU^2: (gate.relu()^2) * up
        let gate_act = gate.execute_relu2()?;
        let inner = self.ffn_sub_norm.forward(&gate_act.elementwise_op(&up, "mul")?)?;
        
        let down_out = self.down_proj.forward(&inner)?;
        // println!("[BitNet] Layer MLP down_proj shape: {:?}", down_out.shape);
        x = residual_mlp.elementwise_op(&down_out, "add")?;
        
        /*
        if x.shape.len() > 1 && x.shape[1] == 1 {
            println!("[BitNet] WARNING! Shape collapse detected at end of MLP residuals: {:?}", x.shape);
        }
        */
        
        Ok((x, k, v))
    }

    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> PyResult<Tensor> {
        // x: [1, seq_len, n_heads, head_dim]
        // cos/sin: [1, seq_len, 1, head_dim] — expand n_heads dim to match x
        let n_heads_x = x.shape[2]; // n_heads or n_kv_heads depending on caller
        let cos_exp = cos.execute_repeat_interleave(n_heads_x, 2)?;
        let sin_exp = sin.execute_repeat_interleave(n_heads_x, 2)?;
        
        let half = self.head_dim / 2;
        let x1 = x.execute_narrow(3, 0, half)?;       // first half of head_dim
        let x2 = x.execute_narrow(3, half, half)?;    // second half of head_dim
        
        // rotate_half: cat(-x2, x1) along dim 3
        let x_rotated = Tensor::execute_cat(&[&x2.execute_neg()?, &x1], 3)?;
        
        // (x * cos) + (x_rotated * sin)
        let term1 = x.elementwise_op(&cos_exp, "mul")?;
        let term2 = x_rotated.elementwise_op(&sin_exp, "mul")?;
        term1.elementwise_op(&term2, "add")
    }

    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> PyResult<Tensor> {
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let (bs, n_heads, q_len, _) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
        let k_len = k.shape[2];

        // q: (1, n_heads, q_len, head_dim)
        // k: (1, n_heads, k_len, head_dim) -> k_t: (1, n_heads, head_dim, k_len)
        let k_t = k.execute_transpose_dims(2, 3)?;
        
        // Fold (bs, n_heads) into a single batch dimension for bmm
        let q_3d = q.execute_reshape(vec![bs * n_heads, q_len, self.head_dim])?;
        let k_t_3d = k_t.execute_reshape(vec![bs * n_heads, self.head_dim, k_len])?;
        
        let logits_3d = q_3d.bmm(&k_t_3d)?;
        let mut logits = logits_3d.execute_reshape(vec![bs, n_heads, q_len, k_len])?
            .scalar_elementwise_op(scale as f64, "mul")?;
        
        // Causal Mask (only if q_len > 1)
        if q_len > 1 {
            let mut mask_data = vec![0.0f32; q_len * k_len];
            for i in 0..q_len {
                for j in 0..k_len {
                    if j > i + (k_len - q_len) {
                        mask_data[i * k_len + j] = -1e10; // -inf
                    }
                }
            }
            let mask = Tensor::new_from_vec(mask_data, vec![1, 1, q_len, k_len], DataType::F32, &q.device, "mask")?;
            let mask_exp = mask.execute_repeat_interleave(bs, 0)?.execute_repeat_interleave(n_heads, 1)?;
            logits = logits.elementwise_op(&mask_exp, "add")?;
        }
        
        let weights = logits.execute_softmax(3, false)?;
        
        let v_3d = v.execute_reshape(vec![bs * n_heads, k_len, self.head_dim])?;
        let weights_3d = weights.execute_reshape(vec![bs * n_heads, q_len, k_len])?;
        let out_3d = weights_3d.bmm(&v_3d)?;
        out_3d.execute_reshape(vec![bs, n_heads, q_len, self.head_dim])
    }
}

pub struct GGUFLoader {
    pub ast: gguf::GGUFFile,
    pub mmap: memmap2::Mmap,
    pub data_offset: usize,
}

impl GGUFLoader {
    pub fn tensor(&self, name: &str) -> PyResult<(&[u8], Vec<usize>, gguf::GGMLType)> {
        let t_info = self.ast.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor {} not found in GGUF", name)))?;
        
        let mut dims = Vec::new();
        for &d in &t_info.dimensions { dims.push(d as usize); }
        
        let mut size_bytes = 1;
        for &d in &dims { size_bytes *= d; }

        let bytes_total = match t_info.tensor_type {
            gguf::GGMLType::F32 => size_bytes * 4,
            gguf::GGMLType::F16 => size_bytes * 2,
            gguf::GGMLType::Q8_0 => size_bytes,
            // MS BitNet I2_S layout packs 4 weights into 1 byte (N / 4 rows mapped internally).
            // Padded with a final 32-byte alignment block containing the f32 quantization scale.
            gguf::GGMLType::I2_S => (size_bytes / 4) + 32,
            _ => size_bytes, 
        };
        
        let start = self.data_offset + t_info.offset as usize;
        let end = start + bytes_total as usize;
        
        Ok((&self.mmap[start..std::cmp::min(end, self.mmap.len())], dims, t_info.tensor_type))
    }
}

#[pyclass]
pub struct BitNetModel {
    pub config: BitNetConfig,
    pub layers: Vec<BitNetLayer>,
    pub embed: Tensor,
    pub norm: BitNetRMSNorm,
    pub lm_head: Tensor,
    pub device: String,
}

#[pymethods]
impl BitNetModel {
    #[new]
    pub fn new(path: &str, device: &str) -> PyResult<Self> {
        // GGUF natively encodes standard configuration headers sequentially; dynamically reading these parameter sets
        // supersedes the requirement for a separate decoupled json tracker document.
        let config = BitNetConfig {
            hidden_size: 2560,
            intermediate_size: 6912,
            num_attention_heads: 20,
            num_key_value_heads: 5,
            num_hidden_layers: 24,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            vocab_size: 128256,
        };

        let model_path = if path.ends_with(".gguf") {
            path.to_string()
        } else {
            format!("{}/ggml-model-i2_s.gguf", path)
        };
        let file = File::open(model_path.clone())?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        
        let (remaining, gguf_ast) = gguf::parser::gguf_file(&mmap)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("GGUF Parse Error: {:?}", e)))?;
            
        // Calculate the base data offset (aligned to 32 bytes by default in GGUF)
        let header_metadata_size = mmap.len() - remaining.len();
        let alignment = 32; // Standard GGUF alignment
        let data_offset = (header_metadata_size + alignment - 1) & !(alignment - 1);
        
        println!("[BitNet] Dynamic GGUF data_offset: {} (Header/KV/Tensors: {})", data_offset, header_metadata_size);

        let tensors = GGUFLoader {
            ast: gguf_ast,
            mmap,
            data_offset,
        };

        let device_str = if device == "vga" || device == "vulkan" { "vulkan" } else { "cpu" };
        
        println!("[BitNet] Loading {} layers on {}...", config.num_hidden_layers, device_str);
        
        let (tok_w, ds, _) = tensors.tensor("token_embd.weight")?;
        // Vocabulary is always significantly larger than the hidden layer dimension structurally.
        let hidden_size = *ds.iter().min().unwrap_or(&2560);
        let vocab_size = *ds.iter().max().unwrap_or(&32000);
        let real_hidden_size = hidden_size;
        
        println!("[BitNet] Detected dimensions: hidden_size={}, vocab_size={}", hidden_size, vocab_size);

        let embed_vec = Tensor::from_raw_bytes(tok_w.to_vec(), vec![vocab_size, hidden_size], DataType::BF16, "cpu")?.to_numpy_f32_vec();
        let embed_tensor = Tensor::new_from_vec(embed_vec, vec![vocab_size, hidden_size], DataType::F32, "cpu", "embed")?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let p = format!("blk.{}", i);
            
            let q_proj = Self::load_bitlinear(&tensors, &format!("{}.attn_q", p), device_str)?;
            let k_proj = Self::load_bitlinear(&tensors, &format!("{}.attn_k", p), device_str)?;
            let v_proj = Self::load_bitlinear(&tensors, &format!("{}.attn_v", p), device_str)?;
            let o_proj = Self::load_bitlinear(&tensors, &format!("{}.attn_output", p), device_str)?;

            let gate_proj = Self::load_bitlinear(&tensors, &format!("{}.ffn_gate", p), device_str)?;
            let up_proj = Self::load_bitlinear(&tensors, &format!("{}.ffn_up", p), device_str)?;
            let down_proj = Self::load_bitlinear(&tensors, &format!("{}.ffn_down", p), device_str)?;

            layers.push(BitNetLayer {
                q_proj, k_proj, v_proj, o_proj,
                gate_proj, up_proj, down_proj,
                input_layernorm: Self::load_norm(&tensors, &format!("{}.attn_norm", p), config.rms_norm_eps)?,
                post_attention_layernorm: Self::load_norm(&tensors, &format!("{}.ffn_norm", p), config.rms_norm_eps)?,
                // Attempt standard sub norm hooks
                attn_sub_norm: Self::load_norm(&tensors, &format!("{}.attn_sub_norm", p), config.rms_norm_eps).unwrap_or_else(|_| BitNetRMSNorm { weight: Tensor::new_from_vec(vec![1.0; real_hidden_size], vec![real_hidden_size], DataType::F32, "cpu", "dummy").unwrap(), eps: 1e-5 }),
                ffn_sub_norm: Self::load_norm(&tensors, &format!("{}.ffn_sub_norm", p), config.rms_norm_eps).unwrap_or_else(|_| BitNetRMSNorm { weight: Tensor::new_from_vec(vec![1.0; real_hidden_size], vec![real_hidden_size], DataType::F32, "cpu", "dummy").unwrap(), eps: 1e-5 }),
                n_heads: config.num_attention_heads,
                n_kv_heads: config.num_key_value_heads,
                head_dim: real_hidden_size / config.num_attention_heads,
            });
        }

        let norm = Self::load_norm(&tensors, "output_norm", config.rms_norm_eps)?;
        
        let lm_head_t = if let Ok((data, _, _)) = tensors.tensor("output.weight") {
            let t = Tensor::from_raw_bytes(data.to_vec(), vec![vocab_size, real_hidden_size], DataType::BF16, "cpu")?.to_numpy_f32_vec();
            Tensor::new_from_vec(t, vec![vocab_size, real_hidden_size], DataType::F32, "cpu", "lm_head")?
        } else {
            embed_tensor.clone()
        };

        Ok(BitNetModel {
            config: BitNetConfig {
                hidden_size: real_hidden_size,
                vocab_size,
                ..config
            },
            layers,
            embed: embed_tensor,
            norm,
            lm_head: lm_head_t,
            device: device_str.to_string(),
        })
    }

    pub fn generate(&self, prompt_ids: Vec<u32>, max_tokens: usize) -> PyResult<Vec<u32>> {
        println!("[BitNet] Generating with prompt size: {}", prompt_ids.len());
        let mut generated = prompt_ids;
        let head_dim = self.config.hidden_size / self.config.num_attention_heads;
        
        // Precompute RoPE (max context 4096)
        println!("[BitNet] Precomputing RoPE...");
        let (cos_table, sin_table) = self.get_rope_cos_sin(4096, head_dim)?;
        let cos_table = cos_table.to(&self.device)?;
        let sin_table = sin_table.to(&self.device)?;
        
        let mut kv_caches: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        
        for _i in 0..max_tokens {
            println!("[BitNet] Generating token {}/{}...", _i + 1, max_tokens);
            let seq_len = generated.len();
            
            // Prefill vs Decode
            let (tokens_to_process, start_pos) = if _i == 0 {
                (generated.clone(), 0)
            } else {
                (vec![*generated.last().unwrap()], seq_len - 1)
            };

            let current_len = tokens_to_process.len();
            let mut x = self.index_select_embed(&tokens_to_process)?;
            
            let cos = cos_table.execute_narrow(1, start_pos, current_len)?;
            let sin = sin_table.execute_narrow(1, start_pos, current_len)?;
            
            for (idx, layer) in self.layers.iter().enumerate() {
                // println!("[BitNet] Processing layer {}/{}...", idx, self.layers.len());
                let (new_x, new_k, new_v) = match &kv_caches[idx] {
                    Some((pk, pv)) => layer.forward(&x, &cos, &sin, Some(pk), Some(pv))?,
                    None => layer.forward(&x, &cos, &sin, None, None)?
                };
                x = new_x;
                kv_caches[idx] = Some((new_k, new_v));
            }
            println!("[BitNet] All layers processed for token {}.", _i);
            
            // Final Norm + LM Head
            let x_last = x.execute_narrow(1, current_len - 1, 1)?.rms_norm(vec![self.config.hidden_size], Some(&self.norm.weight), self.norm.eps)?;
            let x_2d = x_last.execute_reshape(vec![1, self.config.hidden_size])?;
            let logits = x_2d.__matmul__(&self.lm_head.execute_transpose()?)?;
            
            let next_id = logits.execute_argmax(1)?.to_numpy_f32_vec()[0] as u32;
            generated.push(next_id);
            
            // Llama-3 EOS: 128001 / 128009
            if next_id == 128001 || next_id == 128009 { break; }
        }
        
        Ok(generated)
    }
}

impl BitNetModel {
    fn load_bitlinear(tensors: &GGUFLoader, prefix: &str, device: &str) -> PyResult<BitLinear> {
        let (w_data, shape, d_type) = tensors.tensor(&format!("{}.weight", prefix))?;
        println!("[BitNet] Loading {} (type {:?})", prefix, d_type);
        
        // Native MS BitNet GGUF shape is [K, N] in format natively provided by loader.
        let k = shape[0];          // Actual input features (columns)
        let n = shape[1];          // Logical output features (rows)
        let packed_rows = n / 4;   // Groups of 4 output rows tightly packed into bytes

        println!("[BitNet] Loading I2_S Native Block {}: logical {}x{}", prefix, n, k);
        
        // Diagnostic: Print first 16 raw bytes of the tensor data
        print!("[BitNet] Raw Data Snippet ({}): ", prefix);
        for i in 0..min(16, w_data.len()) {
            print!("{:02x} ", w_data[i]);
        }
        println!();

        let weight_bytes = packed_rows * k;  // Native buffer block size without trailing padding

        // MS bitnet default gguf layout (AVX2 ACT_PARALLEL): 
        // 128 elements of a single row are interleaved into 32 bytes.
        // For a 32-byte block, byte `b` contains cols: b (shift 6), b+32 (shift 4), b+64 (shift 2), b+96 (shift 0).
        let q8_len = n * k;
        let mut q8 = vec![0u8; q8_len];
        
        let blocks = q8_len / 128;
        for i in 0..blocks {
            for b in 0..32 {
                let byte_val = w_data[i * 32 + b];
                // Check if this handles the offset mapping correctly.
                // In official I2_S, (val >> 6) & 0x03 result is in {0, 1, 2, 3}
                q8[i * 128 + b]      = (byte_val >> 6) & 0x03;
                q8[i * 128 + b + 32] = (byte_val >> 4) & 0x03;
                q8[i * 128 + b + 64] = (byte_val >> 2) & 0x03;
                q8[i * 128 + b + 96] = byte_val & 0x03;
            }
        }

        // Diagnostic: Print the first few unpacked values to see the range
        print!("[BitNet] Unpacked range sample: ");
        for i in 0..min(16, q8.len()) {
            print!("{} ", q8[i]);
        }
        println!();

        // Our AVX2 kernel expects MSB-first column-major-like packing: 
        // 4 rows of the same column are packed into 1 byte.
        // bits[7:6]=row0, [5:4]=row1, [3:2]=row2, [1:0]=row3
        // ternary mappings for BitNet-2B:
        // GGUF stores {-1, 0, 1} as {1, 2, 3}.
        // Our kernels expect {0, 1, 2} to represent {-1, 0, 1} (with a_sum correction)
        // or raw {-1, 0, 1} if we change the kernel.
        // For compatibility with execute_bit_linear_scalar's (q-1) logic, we need q in {0, 1, 2}.
        let mut rust_packed = vec![0u8; weight_bytes];
        for r_group in 0..packed_rows {
            let r0 = r_group * 4;
            let r1 = r0 + 1;
            let r2 = r0 + 2;
            let r3 = r0 + 3;
            
            for col in 0..k {
                // Subtract 1 from the GGUF value (1,2,3) to get (0,1,2)
                let val0 = q8[r0 * k + col].saturating_sub(1);
                let val1 = q8[r1 * k + col].saturating_sub(1);
                let val2 = q8[r2 * k + col].saturating_sub(1);
                let val3 = q8[r3 * k + col].saturating_sub(1);
                
                // Pack into 2-bit chunks: [r0][r1][r2][r3]
                rust_packed[r_group * k + col] = (val0 << 6) | (val1 << 4) | (val2 << 2) | val3;
            }
        }
        
        // Diagnostic: Print first 16 packed bytes
        print!("[BitNet] Packed Data Snippet ({}): ", prefix);
        for i in 0..min(16, rust_packed.len()) {
            print!("{:02x} ", rust_packed[i]);
        }
        println!();
        println!("[BitNet] weight_data length: {}, expected: {}, delta: {}", w_data.len(), weight_bytes, (w_data.len() as i64) - (weight_bytes as i64));

        let weight = Tensor::from_raw_bytes(rust_packed, vec![n, k], DataType::BitNet2, "cpu")?;
        
        // Extract scale (at exactly weight_bytes offset in the GGUF data block)
        let mut scale_val = 1.0f32;
        if d_type == gguf::GGMLType::I2_S && w_data.len() >= weight_bytes + 4 {
            let s_bytes: [u8; 4] = [
                w_data[weight_bytes], w_data[weight_bytes+1],
                w_data[weight_bytes+2], w_data[weight_bytes+3]
            ];
            scale_val = f32::from_le_bytes(s_bytes);
            println!("[BitNet] Loaded native scale for {}: {} (at offset {})", prefix, scale_val, weight_bytes);
            
            // Diagnostic: Print first few unpacked weights
            if prefix.contains("blk.0.attn_q") {
                print!("[BitNet] Weights Sample ({}): ", prefix);
                for i in 0..10 {
                    let val = q8[i] as i32 - 1; // Map back to ternary {-1, 0, 1}
                    print!("{} ", val);
                }
                println!();
            }
        }
        
        let scale_vec = vec![scale_val; n];
        let scale = Tensor::new_from_vec(scale_vec, vec![n], DataType::F32, "cpu", "scale")?;
        
        // Check for standalone bias mapping cleanly.
        let bias = if let Ok((b_raw, _b_shape, _)) = tensors.tensor(&format!("{}.bias", prefix)) {
            let mut b_vec = Tensor::from_raw_bytes(b_raw.to_vec(), vec![n], DataType::F32, "cpu")?.to_numpy_f32_vec();
            b_vec.resize(n, 0.0);
            Some(Tensor::new_from_vec(b_vec, vec![n], DataType::F32, "cpu", "bias")?)
        } else {
            None
        };

        if device == "vulkan" {
            Ok(BitLinear {
                weight: weight.to_device("vulkan")?,
                scale: scale.to_device("vulkan")?,
                bias: bias.map(|b| b.to_device("vulkan")).transpose()?,
            })
        } else {
            Ok(BitLinear { weight, scale, bias })
        }
    }

    fn load_norm(tensors: &GGUFLoader, prefix: &str, eps: f32) -> PyResult<BitNetRMSNorm> {
        let (w_data, w_shape, _d_type) = tensors.tensor(&format!("{}.weight", prefix))?;
        // Norm weights are F32 in GGUF
        let weight = Tensor::from_raw_bytes(w_data.to_vec(), vec![w_shape[0]], DataType::F32, "cpu")?;
        let w_f32 = weight.to_numpy_f32_vec();
        let weight_f32 = Tensor::new_from_vec(w_f32, vec![w_shape[0]], DataType::F32, "cpu", "norm_w")?;
        Ok(BitNetRMSNorm { weight: weight_f32, eps })
    }

    fn get_rope_cos_sin(&self, max_seq: usize, dim: usize) -> PyResult<(Tensor, Tensor)> {
        let theta = self.config.rope_theta;
        let mut inv_freq = vec![0.0f32; dim / 2];
        for i in 0..(dim / 2) {
            inv_freq[i] = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        }
        
        let mut cos_data = vec![0.0f32; max_seq * dim];
        let mut sin_data = vec![0.0f32; max_seq * dim];
        
        for t in 0..max_seq {
            for i in 0..(dim / 2) {
                let freq = (t as f32) * inv_freq[i];
                let c = freq.cos();
                let s = freq.sin();
                cos_data[t * dim + i] = c;
                cos_data[t * dim + i + dim/2] = c;
                sin_data[t * dim + i] = s;
                sin_data[t * dim + i + dim/2] = s;
            }
        }
        
        let cos = Tensor::new_from_vec(cos_data, vec![1, max_seq, 1, dim], DataType::F32, "cpu", "cos")?;
        let sin = Tensor::new_from_vec(sin_data, vec![1, max_seq, 1, dim], DataType::F32, "cpu", "sin")?;
        Ok((cos, sin))
    }

    fn index_select_embed(&self, ids: &[u32]) -> PyResult<Tensor> {
        let ids_vec: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
        let ids_tensor = Tensor::new_from_vec(ids_vec, vec![ids.len()], DataType::F32, "cpu", "indices")?;
        
        let res = self.embed.index_select(0, &ids_tensor)?;
        let res_3d = res.execute_reshape(vec![1, ids.len(), self.config.hidden_size])?;
        res_3d.to(&self.device)
    }
}
