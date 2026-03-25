use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;
use crate::tensor::{Tensor, DataType, Storage};
use rayon::prelude::*;

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
        // x must be Int8 for BitLinear
        // Typically x is normalized and then quantized to Int8 per-token
        let x_q = self.quantize_activation(x)?;
        Tensor::execute_bit_linear(&x_q, &self.weight, &self.scale, self.bias.as_ref())
    }

    fn quantize_activation(&self, x: &Tensor) -> PyResult<Tensor> {
        // symmetric per-token quantization (absmax)
        // For now, doing it on CPU/Rayon or via a shader if available
        // VNN currently doesn't have a fused quant kernel in execute_bit_linear,
        // so we do it here.
        
        let shape = x.shape.clone();
        let m = shape[0];
        let k = shape[1];
        
        // This is a slow path if it's not a kernel. 
        // TODO: Move per-token quantization into a VNN kernel.
        let mut res = Tensor::new_zeros(shape, DataType::Int8, &x.device)?;
        
        if x.device == "cpu" {
             let (src, _) = x.get_slice_raw_f32();
             let (dst, _) = res.get_slice_raw_mut_i8();
             
             dst.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
                 let src_row = &src[i * k .. (i+1) * k];
                 let mut max_abs = 0.0f32;
                 for &val in src_row {
                     let abs = val.abs();
                     if abs > max_abs { max_abs = abs; }
                 }
                 let scale = 127.0 / max_abs.max(1e-5);
                 for j in 0..k {
                     row[j] = (src_row[j] * scale).round().clamp(-128.0, 127.0) as i8;
                 }
             });
        }
        
        Ok(res)
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
        
        // 1. Attention
        let q = self.q_proj.forward(&x_norm)?.execute_reshape(vec![1, 0, self.n_heads, self.head_dim])?;
        let k = self.k_proj.forward(&x_norm)?.execute_reshape(vec![1, 0, self.n_kv_heads, self.head_dim])?;
        let v = self.v_proj.forward(&x_norm)?.execute_reshape(vec![1, 0, self.n_kv_heads, self.head_dim])?;
        
        // RoPE
        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;

        // KV Cache update
        let (k, v) = match (past_k, past_v) {
            (Some(pk), Some(pv)) => {
                let new_k = Tensor::execute_cat(&[pk, &k], 1)?;
                let new_v = Tensor::execute_cat(&[pv, &v], 1)?;
                (new_k, new_v)
            },
            _ => (k, v)
        };
        
        // GQA Repeat if needed
        let (k_rep, v_rep) = if self.n_kv_heads < self.n_heads {
            let kr = k.execute_repeat_interleave(self.n_heads / self.n_kv_heads, 2)?;
            let vr = v.execute_repeat_interleave(self.n_heads / self.n_kv_heads, 2)?;
            (kr, vr)
        } else {
            (k.clone(), v.clone())
        };
        
        // SDPA: (bs, n_heads, seq_len, head_dim)
        let q_tp = q.execute_transpose_dims(1, 2)?;
        let k_tp = k_rep.execute_transpose_dims(1, 2)?;
        let v_tp = v_rep.execute_transpose_dims(1, 2)?;
        
        let mut attn_out = self.scaled_dot_product_attention(&q_tp, &k_tp, &v_tp)?;
        
        // Transpose back: (bs, seq_len, n_heads, head_dim)
        attn_out = attn_out.execute_transpose_dims(1, 2)?;
        let attn_out_flat = attn_out.execute_reshape(vec![1, 0, self.n_heads * self.head_dim])?;
        
        // SubLN + Output Projection
        let attn_norm = self.attn_sub_norm.forward(&attn_out_flat)?;
        let mut x = residual.elementwise_op(&self.o_proj.forward(&attn_norm)?, "add")?;
        
        // 2. MLP
        let residual_mlp = x.clone();
        let x_norm_mlp = self.post_attention_layernorm.forward(&x)?;
        
        let gate = self.gate_proj.forward(&x_norm_mlp)?;
        let up = self.up_proj.forward(&x_norm_mlp)?;
        
        // ReLU^2: (gate.relu()^2) * up
        let gate_act = gate.relu()?.execute_pow(2.0)?;
        let inner = self.ffn_sub_norm.forward(&gate_act.elementwise_op(&up, "mul")?)?;
        
        x = residual_mlp.elementwise_op(&self.down_proj.forward(&inner)?, "add")?;
        
        Ok((x, k, v))
    }

    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> PyResult<Tensor> {
        // x: (1, seq_len, n_heads, head_dim)
        // cos, sin: (1, seq_len, 1, head_dim)
        // rotate_half: cat(-x2, x1)
        let half = self.head_dim / 2;
        let x1 = x.execute_narrow(3, 0, half)?;
        let x2 = x.execute_narrow(3, half, half)?;
        
        let x_rotated = Tensor::execute_cat(&[&x2.execute_neg()?, &x1], 3)?;
        
        // (x * cos) + (x_rotated * sin)
        let term1 = x.elementwise_op(cos, "mul")?;
        let term2 = x_rotated.elementwise_op(sin, "mul")?;
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
        // ... (loading logic remains same)
        let config_path = format!("{}/config.json", path);
        let mut config_file = File::open(config_path)?;
        let mut config_str = String::new();
        config_file.read_to_string(&mut config_str)?;
        let config: BitNetConfig = serde_json::from_str(&config_str).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let model_path = format!("{}/model.safetensors", path);
        let bytes = std::fs::read(model_path)?;
        let tensors = SafeTensors::deserialize(&bytes).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let device_str = device.to_string();
        
        // Loader logic...
        // For each layer, unpack and repack
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{}", i);
            layers.push(BitNetLayer {
                q_proj: Self::load_bitlinear(&tensors, &format!("{}.self_attn.q_proj", p), config.hidden_size, config.hidden_size, device)?,
                k_proj: Self::load_bitlinear(&tensors, &format!("{}.self_attn.k_proj", p), (config.num_key_value_heads * config.hidden_size) / config.num_attention_heads, config.hidden_size, device)?,
                v_proj: Self::load_bitlinear(&tensors, &format!("{}.self_attn.v_proj", p), (config.num_key_value_heads * config.hidden_size) / config.num_attention_heads, config.hidden_size, device)?,
                o_proj: Self::load_bitlinear(&tensors, &format!("{}.self_attn.o_proj", p), config.hidden_size, config.hidden_size, device)?,
                gate_proj: Self::load_bitlinear(&tensors, &format!("{}.mlp.gate_proj", p), config.intermediate_size, config.hidden_size, device)?,
                up_proj: Self::load_bitlinear(&tensors, &format!("{}.mlp.up_proj", p), config.intermediate_size, config.hidden_size, device)?,
                down_proj: Self::load_bitlinear(&tensors, &format!("{}.mlp.down_proj", p), config.hidden_size, config.intermediate_size, device)?,
                input_layernorm: Self::load_norm(&tensors, &format!("{}.input_layernorm", p), config.rms_norm_eps)?,
                post_attention_layernorm: Self::load_norm(&tensors, &format!("{}.post_attention_layernorm", p), config.rms_norm_eps)?,
                attn_sub_norm: Self::load_norm(&tensors, &format!("{}.self_attn.attn_sub_norm", p), config.rms_norm_eps)?,
                ffn_sub_norm: Self::load_norm(&tensors, &format!("{}.mlp.ffn_sub_norm", p), config.rms_norm_eps)?,
                n_heads: config.num_attention_heads,
                n_kv_heads: config.num_key_value_heads,
                head_dim: config.hidden_size / config.num_attention_heads,
            });
        }

        let embed_data = tensors.tensor("model.embed_tokens.weight").map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let embed = Tensor::from_raw_bytes(embed_data.data().to_vec(), vec![config.vocab_size, config.hidden_size], DataType::F32, "cpu")?;

        let norm = Self::load_norm(&tensors, "model.norm", config.rms_norm_eps)?;
        
        let lm_head = if let Ok(data) = tensors.tensor("lm_head.weight") {
            Tensor::from_raw_bytes(data.data().to_vec(), vec![config.vocab_size, config.hidden_size], DataType::F32, "cpu")?
        } else {
            embed.clone()
        };

        Ok(BitNetModel {
            config,
            layers,
            embed,
            norm,
            lm_head,
            device: device_str,
        })
    }

    pub fn generate(&self, prompt_ids: Vec<u32>, max_tokens: usize) -> PyResult<Vec<u32>> {
        let mut generated = prompt_ids;
        let head_dim = self.config.hidden_size / self.config.num_attention_heads;
        
        // Precompute RoPE (max context 4096)
        let (cos_table, sin_table) = self.get_rope_cos_sin(4096, head_dim)?;
        
        let mut kv_caches: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        
        for i in 0..max_tokens {
            let seq_len = generated.len();
            
            // 1. Determine if we are in prefill or decode
            // Prefill: first iteration, process all prompt tokens
            // Decode: subsequent iterations, process only the last token
            let (tokens_to_process, start_pos) = if i == 0 {
                (generated.clone(), 0)
            } else {
                (vec![*generated.last().unwrap()], seq_len - 1)
            };

            let current_len = tokens_to_process.len();
            
            // 2. Embedding Lookup
            let mut x = self.index_select_embed(&tokens_to_process)?;
            
            // 3. RoPE Slicing
            let cos = cos_table.execute_narrow(1, start_pos, current_len)?;
            let sin = sin_table.execute_narrow(1, start_pos, current_len)?;
            
            // 4. Layers with KV Cache
            for (idx, layer) in self.layers.iter().enumerate() {
                let (new_x, new_k, new_v) = match &kv_caches[idx] {
                    Some((pk, pv)) => layer.forward(&x, &cos, &sin, Some(pk), Some(pv))?,
                    None => layer.forward(&x, &cos, &sin, None, None)?
                };
                x = new_x;
                kv_caches[idx] = Some((new_k, new_v));
            }
            
            // 5. Final Norm + LM Head (last token only)
            let x_last = x.execute_narrow(1, current_len - 1, 1)?.py_rms_norm(vec![self.config.hidden_size], Some(&self.norm.weight), self.norm.eps)?;
            let logits = x_last.__matmul__(&self.lm_head.execute_transpose()?)?;
            
            // 6. Argmax
            let next_id = logits.execute_argmax(2)?.to_numpy_f32_vec()[0] as u32;
            generated.push(next_id);
            
            // TODO: Stop on EOS
        }
        
        Ok(generated)
    }
}

impl BitNetModel {

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
        
        let cos = Tensor::new_from_vec(cos_data, vec![1, max_seq, 1, dim], DataType::F32, &self.device, "cos")?;
        let sin = Tensor::new_from_vec(sin_data, vec![1, max_seq, 1, dim], DataType::F32, &self.device, "sin")?;
        Ok((cos, sin))
    }

    fn index_select_embed(&self, ids: &[u32]) -> PyResult<Tensor> {
        let n = ids.len();
        let h = self.config.hidden_size;
        let mut data = vec![0.0f32; n * h];
        
        let (embed_raw, _) = self.embed.get_slice_raw_f32();
        
        data.par_chunks_mut(h).enumerate().for_each(|(i, row)| {
            let id = ids[i] as usize;
            if id < self.config.vocab_size {
                let start = id * h;
                row.copy_from_slice(&embed_raw[start..start + h]);
            }
        });
        
        Tensor::new_from_vec(data, vec![1, n, h], DataType::F32, &self.device, "embed_lookup")
    }

    fn load_bitlinear(tensors: &SafeTensors, prefix: &str, m: usize, k: usize, device: &str) -> PyResult<BitLinear> {
        let w_data = tensors.tensor(&format!("{}.weight", prefix)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let weight = Tensor::from_raw_bytes(w_data.data().to_vec(), vec![m, k], DataType::BitNet2, "cpu")?;
        
        let s_data = tensors.tensor(&format!("{}.weight_scale", prefix)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let scale = Tensor::from_raw_bytes(s_data.data().to_vec(), vec![m], DataType::F32, "cpu")?;
        
        let bias = if let Ok(data) = tensors.tensor(&format!("{}.bias", prefix)) {
            Some(Tensor::from_raw_bytes(data.data().to_vec(), vec![m], DataType::F32, "cpu")?)
        } else {
            None
        };

        if device == "vulkan" {
             Ok(BitLinear {
                 weight: weight.to_device(device)?,
                 scale: scale.to_device(device)?,
                 bias: if let Some(b) = bias { Some(b.to_device(device)?) } else { None },
             })
        } else {
             Ok(BitLinear { weight, scale, bias })
        }
    }

    fn load_norm(tensors: &SafeTensors, prefix: &str, eps: f32) -> PyResult<BitNetRMSNorm> {
        let w_data = tensors.tensor(&format!("{}.weight", prefix)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let weight = Tensor::from_raw_bytes(w_data.data().to_vec(), vec![w_data.shape()[0]], DataType::F32, "cpu")?;
        Ok(BitNetRMSNorm { weight, eps })
    }
}
