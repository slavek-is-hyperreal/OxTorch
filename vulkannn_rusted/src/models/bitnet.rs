use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;
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
        // Weight shape is [N/4, K] (packed), logical N = weight.shape[0] * 4
        let logical_n = self.weight.shape[0] * 4;
        let mut y = Tensor::execute_bit_linear_with_n(&x_q, &self.weight, logical_n, &self.scale, self.bias.as_ref())?;
        // Apply activation scale
        y = y.execute_mul_broadcast(&s_a)?;
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
        
        let seq_len = x_norm.shape[0]; // actual tokens this forward pass
        // 1. Attention — reshape to [1, seq_len, n_heads, head_dim] (batch=1)
        let q = self.q_proj.forward(&x_norm)?.execute_reshape(vec![1, seq_len, self.n_heads, self.head_dim])?;
        let k = self.k_proj.forward(&x_norm)?.execute_reshape(vec![1, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = self.v_proj.forward(&x_norm)?.execute_reshape(vec![1, seq_len, self.n_kv_heads, self.head_dim])?;
        
        // RoPE
        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;

        // KV Cache update — cat along seq dim (1)
        let (k, v) = match (past_k, past_v) {
            (Some(pk), Some(pv)) => {
                let new_k = Tensor::execute_cat(&[pk, &k], 1)?;
                let new_v = Tensor::execute_cat(&[pv, &v], 1)?;
                (new_k, new_v)
            },
            _ => (k, v)
        };
        
        // GQA Repeat if needed: k/v are [1, kv_seq, n_kv_heads, head_dim]
        let (k_rep, v_rep) = if self.n_kv_heads < self.n_heads {
            let kr = k.execute_repeat_interleave(self.n_heads / self.n_kv_heads, 2)?;
            let vr = v.execute_repeat_interleave(self.n_heads / self.n_kv_heads, 2)?;
            (kr, vr)
        } else {
            (k.clone(), v.clone())
        };
        
        // SDPA: transpose to (bs=1, n_heads, seq, head_dim)
        let q_tp = q.execute_transpose_dims(1, 2)?;
        let k_tp = k_rep.execute_transpose_dims(1, 2)?;
        let v_tp = v_rep.execute_transpose_dims(1, 2)?;
        
        let mut attn_out = self.scaled_dot_product_attention(&q_tp, &k_tp, &v_tp)?;
        
        // Transpose back: (1, n_heads, seq, head_dim) → (1, seq, n_heads, head_dim)
        attn_out = attn_out.execute_transpose_dims(1, 2)?;
        // Flatten to [seq_len, n_heads*head_dim]
        let attn_out_flat = attn_out.execute_reshape(vec![seq_len, self.n_heads * self.head_dim])?;
        
        // SubLN + Output Projection
        let attn_norm = self.attn_sub_norm.forward(&attn_out_flat)?;
        let mut x = residual.elementwise_op(&self.o_proj.forward(&attn_norm)?, "add")?;
        
        // 2. MLP
        let residual_mlp = x.clone();
        let x_norm_mlp = self.post_attention_layernorm.forward(&x)?;
        
        let gate = self.gate_proj.forward(&x_norm_mlp)?;
        let up = self.up_proj.forward(&x_norm_mlp)?;
        
        // ReLU^2: (gate.relu()^2) * up
        let gate_act = gate.execute_relu2()?;
        let inner = self.ffn_sub_norm.forward(&gate_act.elementwise_op(&up, "mul")?)?;
        
        x = residual_mlp.elementwise_op(&self.down_proj.forward(&inner)?, "add")?;
        
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
        let (_bs, _n_heads, q_len, _) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
        let k_len = k.shape[2];

        // q: (1, n_heads, q_len, head_dim)
        // k: (1, n_heads, k_len, head_dim) -> k_t: (1, n_heads, head_dim, k_len)
        let k_t = k.execute_transpose_dims(2, 3)?;
        let mut logits = q.bmm(&k_t)?.scalar_elementwise_op(scale as f64, "mul")?;
        
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
            logits = logits.elementwise_op(&mask, "add")?;
        }
        
        let weights = logits.execute_softmax(3, false)?;
        weights.bmm(v)
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
        let config_path = format!("{}/config.json", path);
        let mut config_file = File::open(config_path)?;
        let mut config_str = String::new();
        config_file.read_to_string(&mut config_str)?;
        let config: BitNetConfig = serde_json::from_str(&config_str).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let model_path = format!("{}/model.safetensors", path);
        // Map file for efficiency
        let file = File::open(model_path)?;
        let buffer = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&buffer).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let device_str = if device == "vga" || device == "vulkan" { "vulkan" } else { "cpu" };
        
        println!("[BitNet] Loading {} layers on {}...", config.num_hidden_layers, device_str);
        
        // Dynamically determine dimensions from embed tensor
        let embed_data = tensors.tensor("model.embed_tokens.weight").map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let real_hidden_size = embed_data.shape()[1];
        let vocab_size = embed_data.shape()[0];
        println!("[BitNet] Detected dimensions: hidden_size={}, vocab_size={}", real_hidden_size, vocab_size);

        let embed_vec = Tensor::from_raw_bytes(embed_data.data().to_vec(), vec![vocab_size, real_hidden_size], DataType::BF16, "cpu")?.to_numpy_f32_vec();
        let embed_tensor = Tensor::new_from_vec(embed_vec, vec![vocab_size, real_hidden_size], DataType::F32, "cpu", "embed")?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{}", i);
            
            // Proj sizes are determined by safe-open in load_bitlinear
            let q_proj = Self::load_bitlinear(&tensors, &format!("{}.self_attn.q_proj", p), device_str)?;
            let k_proj = Self::load_bitlinear(&tensors, &format!("{}.self_attn.k_proj", p), device_str)?;
            let v_proj = Self::load_bitlinear(&tensors, &format!("{}.self_attn.v_proj", p), device_str)?;
            let o_proj = Self::load_bitlinear(&tensors, &format!("{}.self_attn.o_proj", p), device_str)?;

            let gate_proj = Self::load_bitlinear(&tensors, &format!("{}.mlp.gate_proj", p), device_str)?;
            let up_proj = Self::load_bitlinear(&tensors, &format!("{}.mlp.up_proj", p), device_str)?;
            let down_proj = Self::load_bitlinear(&tensors, &format!("{}.mlp.down_proj", p), device_str)?;

            layers.push(BitNetLayer {
                q_proj, k_proj, v_proj, o_proj,
                gate_proj, up_proj, down_proj,
                input_layernorm: Self::load_norm(&tensors, &format!("{}.input_layernorm", p), config.rms_norm_eps)?,
                post_attention_layernorm: Self::load_norm(&tensors, &format!("{}.post_attention_layernorm", p), config.rms_norm_eps)?,
                attn_sub_norm: Self::load_norm(&tensors, &format!("{}.self_attn.attn_sub_norm", p), config.rms_norm_eps)?,
                ffn_sub_norm: Self::load_norm(&tensors, &format!("{}.mlp.ffn_sub_norm", p), config.rms_norm_eps)?,
                n_heads: config.num_attention_heads,
                n_kv_heads: config.num_key_value_heads,
                head_dim: real_hidden_size / config.num_attention_heads,
            });
        }

        let norm = Self::load_norm(&tensors, "model.norm", config.rms_norm_eps)?;
        
        let lm_head_t = if let Ok(data) = tensors.tensor("lm_head.weight") {
            let t = Tensor::from_raw_bytes(data.data().to_vec(), vec![vocab_size, real_hidden_size], DataType::BF16, "cpu")?.to_numpy_f32_vec();
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
        let mut generated = prompt_ids;
        let head_dim = self.config.hidden_size / self.config.num_attention_heads;
        
        // Precompute RoPE (max context 4096)
        let (cos_table, sin_table) = self.get_rope_cos_sin(4096, head_dim)?;
        let cos_table = cos_table.to(&self.device)?;
        let sin_table = sin_table.to(&self.device)?;
        
        let mut kv_caches: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        
        for _i in 0..max_tokens {
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
                let (new_x, new_k, new_v) = match &kv_caches[idx] {
                    Some((pk, pv)) => layer.forward(&x, &cos, &sin, Some(pk), Some(pv))?,
                    None => layer.forward(&x, &cos, &sin, None, None)?
                };
                x = new_x;
                kv_caches[idx] = Some((new_k, new_v));
            }
            
            // Final Norm + LM Head
            let x_last = x.execute_narrow(1, current_len - 1, 1)?.rms_norm(vec![self.config.hidden_size], Some(&self.norm.weight), self.norm.eps)?;
            let logits = x_last.__matmul__(&self.lm_head.execute_transpose()?)?;
            
            let next_id = logits.execute_argmax(2)?.to_numpy_f32_vec()[0] as u32;
            generated.push(next_id);
            
            // Llama-3 EOS: 128001 / 128009
            if next_id == 128001 || next_id == 128009 { break; }
        }
        
        Ok(generated)
    }
}

impl BitNetModel {
    fn load_bitlinear(tensors: &SafeTensors, prefix: &str, device: &str) -> PyResult<BitLinear> {
        let w_data = tensors.tensor(&format!("{}.weight", prefix)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let shape = w_data.shape();
        
        // Weight is stored with PACKED shape [N/4, K] matching actual byte count
        // Kernel computes logical N = shape[0] * 4 internally via execute_bit_linear_with_n
        let packed_rows = shape[0];  // = N/4
        let n = packed_rows * 4;     // logical output channels
        let k = shape[1];

        println!("[BitNet] Loading {}: packed [{}, {}] -> logical {}x{}", prefix, packed_rows, k, n, k);

        // Load raw bytes with packed shape [N/4, K] — byte count matches exactly
        let weight = Tensor::from_raw_bytes(w_data.data().to_vec(), vec![packed_rows, k], DataType::BitNet2, "cpu")?;
        
        let s_data = tensors.tensor(&format!("{}.weight_scale", prefix)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let s_raw = s_data.data();

        let s_dtype = if s_raw.len() >= n * 4 { DataType::F32 }
                     else if s_raw.len() >= n * 2 { DataType::BF16 }
                     else if s_raw.len() == 4 { DataType::F32 }
                     else { DataType::BF16 }; // shape=[1] BF16 = 2 bytes

        let s_logical_shape = if s_raw.len() >= n * 2 { vec![n] } else { vec![1] };
        let mut scale_vec = Tensor::from_raw_bytes(s_raw.to_vec(), s_logical_shape, s_dtype, "cpu")?.to_numpy_f32_vec();
        
        if scale_vec.len() == 1 {
            scale_vec = vec![scale_vec[0]; n];
        } else if scale_vec.len() < n {
             scale_vec.resize(n, 1.0);
        }
        let scale = Tensor::new_from_vec(scale_vec, vec![n], DataType::F32, "cpu", "scale")?;
        
        let bias = if let Ok(data) = tensors.tensor(&format!("{}.bias", prefix)) {
            let b_raw = data.data();
            let b_dtype = if b_raw.len() >= n * 4 { DataType::F32 } else { DataType::BF16 };
            let b_shape = if b_raw.len() >= n * 2 { vec![n] } else { vec![1] };
            let mut b_vec = Tensor::from_raw_bytes(b_raw.to_vec(), b_shape, b_dtype, "cpu")?.to_numpy_f32_vec();
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

    fn load_norm(tensors: &SafeTensors, prefix: &str, eps: f32) -> PyResult<BitNetRMSNorm> {
        let w_data = tensors.tensor(&format!("{}.weight", prefix)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        // Norm weights are BF16 in the safetensors file, convert to F32
        let weight = Tensor::from_raw_bytes(w_data.data().to_vec(), vec![w_data.shape()[0]], DataType::BF16, "cpu")?;
        let w_f32 = weight.to_numpy_f32_vec();
        let weight_f32 = Tensor::new_from_vec(w_f32, vec![w_data.shape()[0]], DataType::F32, "cpu", "norm_w")?;
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
        res.to(&self.device)
    }
}
