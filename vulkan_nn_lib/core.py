import taichi as ti
import numpy as np
import os

# Initialize Taichi with Vulkan
# We assume the user has a Vulkan-capable GPU. 
# We'll try Vulkan first, fallback to CPU if needed.
try:
    ti.init(arch=ti.vulkan)
except:
    ti.init(arch=ti.cpu)

@ti.kernel
def k_zero(X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        X[i] = 0.0

@ti.kernel
def k_copy(Src: ti.types.ndarray(), Dst: ti.types.ndarray(), Total: int):
    for i in range(Total):
        Dst[i] = Src[i]

@ti.kernel
def k_gelu_tanh(X: ti.types.ndarray(), Total: int):
    # GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    for i in range(Total):
        x = X[i]
        X[i] = 0.5 * x * (1.0 + ti.tanh(0.79788456 * (x + 0.044715 * x**3)))

@ti.kernel
def k_repeat_kv(Src: ti.types.ndarray(), Dst: ti.types.ndarray(), B: int, L: int, H_kv: int, D: int, factor: int):
    # Expand KV heads directly on GPU: factor = H_q // H_kv
    for b, l, h_kv, i, f in ti.ndrange(B, L, H_kv, D, factor):
        src_idx = ((b * L + l) * H_kv + h_kv) * D + i
        dst_idx = ((b * L + l) * (H_kv * factor) + (h_kv * factor + f)) * D + i
        Dst[dst_idx] = Src[src_idx]

@ti.kernel
def k_altup_predict(H: ti.types.ndarray(), Coefs: ti.types.ndarray(), Out: ti.types.ndarray(),
                    S: int, B: int, L: int, D: int):
    # Coefs is [B, L, S, S]. Out[s, b, l, d] = H[s, b, l, d] + sum_{s2} H[s2, b, l, d] * clamp(Coefs[b, l, s, s2], -120, 120)
    for s, b, l, d in ti.ndrange(S, B, L, D):
        acc = 0.0
        for s2 in range(S):
            src_val = H[((s2 * B + b) * L + l) * D + d]
            c_val = Coefs[((b * L + l) * S + s) * S + s2]
            if c_val > 120.0: c_val = 120.0
            if c_val < -120.0: c_val = -120.0
            acc += src_val * c_val
        Out[((s * B + b) * L + l) * D + d] = H[((s * B + b) * L + l) * D + d] + acc

@ti.kernel
def k_altup_correct(P: ti.types.ndarray(), Activated: ti.types.ndarray(), Coefs: ti.types.ndarray(),
                    S: int, B: int, L: int, D: int, active_idx: int):
    # Coefs is [B, L, S]. P[s, b, l, d] += (Activated[b, l, d] - P[active_idx, b, l, d]) * clamp(Coefs[b, l, s], -120, 120)
    for s, b, l, d in ti.ndrange(S, B, L, D):
        act_val = Activated[(b * L + l) * D + d]
        pred_active_val = P[((active_idx * B + b) * L + l) * D + d]
        innovation = act_val - pred_active_val
        c_val = Coefs[(b * L + l) * S + s]
        if c_val > 120.0: c_val = 120.0
        if c_val < -120.0: c_val = -120.0
        P[((s * B + b) * L + l) * D + d] += innovation * c_val

@ti.kernel
def k_magnitude_norm(X: ti.types.ndarray(), TargetX: ti.types.ndarray(), S_idx: int, T_idx: int, 
                     B: int, L: int, D: int, eps: float):
    # Scale slice S_idx of X such that its mean squared magnitude matches slice T_idx of TargetX
    for b, l in ti.ndrange(B, L):
        target_sum_sq = 0.0
        for d in range(D):
            val = TargetX[((T_idx * B + b) * L + l) * D + d]
            target_sum_sq += val * val
        target_mag = ti.sqrt(target_sum_sq / D + eps)
        
        curr_sum_sq = 0.0
        for d in range(D):
            val = X[((S_idx * B + b) * L + l) * D + d]
            curr_sum_sq += val * val
        curr_mag = ti.sqrt(curr_sum_sq / D + eps)
        
        factor = target_mag / curr_mag
        for d in range(D):
            X[((S_idx * B + b) * L + l) * D + d] *= factor

@ti.kernel
def k_gaussian_topk(X: ti.types.ndarray(), Total: int, N: int, std_multiplier: float):
    # X_i = relu(X_i - (mean + std * multiplier)) per segment N
    for i in range(Total // N):
        mean = 0.0
        sq_mean = 0.0
        for j in range(N):
            val = X[i * N + j]
            mean += val
            sq_mean += val * val
        mean /= N
        std = ti.sqrt(ti.max(0.0, sq_mean / N - mean * mean))
        cutoff = mean + std * std_multiplier
        for j in range(N):
            val = X[i * N + j] - cutoff
            if val < 0.0: val = 0.0
            X[i * N + j] = val

@ti.kernel
def k_extract_slice(Src: ti.types.ndarray(), Dst: ti.types.ndarray(), S_idx: int, B: int, L: int, D: int):
    for b, l, d in ti.ndrange(B, L, D):
        Dst[(b * L + l) * D + d] = Src[((S_idx * B + b) * L + l) * D + d]

@ti.kernel
def k_add_to_slices(Src: ti.types.ndarray(), Dst: ti.types.ndarray(), start_s: int, end_s: int, B: int, L: int, D: int):
    for s, b, l, d in ti.ndrange(ti.range(start_s, end_s), B, L, D):
        Dst[((s * B + b) * L + l) * D + d] += Src[(b * L + l) * D + d]

@ti.kernel
def k_rope(X: ti.types.ndarray(), cos: ti.types.ndarray(), sin: ti.types.ndarray(), 
           B: int, L: int, H: int, D: int, pos_offset: int):
    # Rotary Positional Embedding with absolute position support
    for b, l, h, i in ti.ndrange(B, L, H, D // 2):
        idx1 = ((b * L + l) * H + h) * D + i
        idx2 = ((b * L + l) * H + h) * D + i + D // 2
        
        v1 = X[idx1]
        v2 = X[idx2]
        
        abs_pos = pos_offset + l
        c = cos[abs_pos * (D // 2) + i]
        s = sin[abs_pos * (D // 2) + i]
        
        X[idx1] = v1 * c - v2 * s
        X[idx2] = v1 * s + v2 * c

@ti.kernel
def k_matmul(A: ti.types.ndarray(), B: ti.types.ndarray(), C: ti.types.ndarray(), M: int, N: int, K: int):
    for i, j in ti.ndrange(M, N):
        acc = 0.0
        for k in range(K):
            acc += A[i * K + k] * B[k * N + j]
        C[i * N + j] = acc

@ti.kernel
def k_matmul_tiled(A: ti.types.ndarray(), B_tile: ti.types.ndarray(), C: ti.types.ndarray(), 
                   M: int, N: int, K: int, n_offset: int, n_tile: int, stride: int):
    # A is (M, K), B_tile is (K, stride) padded, C is (M, N)
    for i, j_tile in ti.ndrange(M, n_tile):
        j = n_offset + j_tile
        acc = 0.0
        for k in range(K):
            acc += A[i * K + k] * B_tile[k * stride + j_tile]
        C[i * N + j] += acc

@ti.kernel
def k_add_bias(X: ti.types.ndarray(), bias: ti.types.ndarray(), M: int, N: int):
    for i, j in ti.ndrange(M, N):
        X[i * N + j] += bias[j]

@ti.kernel
def k_relu_1d(X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        if X[i] < 0: X[i] = 0.0

@ti.kernel
def k_leaky_relu_1d(X: ti.types.ndarray(), Total: int, alpha: float):
    for i in range(Total):
        if X[i] < 0: X[i] *= alpha

@ti.kernel
def k_softmax_1d(X: ti.types.ndarray(), M: int, N: int):
    for i in range(M):
        max_val = -1e30
        for j in range(N):
            val = X[i * N + j]
            if val > max_val: max_val = val
        
        expsum = 0.0
        for j in range(N):
            val = ti.exp(X[i * N + j] - max_val)
            X[i * N + j] = val
            expsum += val
            
        for j in range(N):
            X[i * N + j] /= expsum

@ti.kernel
def k_rmsnorm_1d(X: ti.types.ndarray(), W: ti.types.ndarray(), M: int, N: int, eps: float):
    for i in range(M):
        rms = 0.0
        for j in range(N):
            val = X[i * N + j]
            rms += val * val
        rms = ti.sqrt(rms / N + eps)
        for j in range(N):
            X[i * N + j] = (X[i * N + j] / rms) * W[j]

@ti.kernel
def k_silu_1d(X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        val = X[i]
        X[i] = val * (1.0 / (1.0 + ti.exp(-val)))

@ti.kernel
def k_embedding_1d(indices: ti.types.ndarray(), weight: ti.types.ndarray(), out: ti.types.ndarray(), B: int, L: int, D: int):
    for b, l in ti.ndrange(B, L):
        idx = ti.cast(indices[b * L + l], ti.i32)
        for d in range(D):
            out[(b * L + l) * D + d] = weight[idx * D + d]

@ti.kernel
def k_conv2d_1d(X: ti.types.ndarray(), W: ti.types.ndarray(), B: ti.types.ndarray(), Out: ti.types.ndarray(),
                batch: int, in_c: int, out_c: int, h: int, w: int, kh: int, kw: int):
    oh, ow = h - kh + 1, w - kw + 1
    for b, oc, i, j in ti.ndrange(batch, out_c, oh, ow):
        acc = B[oc]
        for ic, ky, kx in ti.ndrange(in_c, kh, kw):
            idx_x = (((b * in_c + ic) * h + (i + ky)) * w + (j + kx))
            acc += X[idx_x] * W[((oc * in_c + ic) * kh + ky) * kw + kx]
        Out[((b * out_c + oc) * oh + i) * ow + j] = acc

@ti.kernel
def k_pool2d_1d(X: ti.types.ndarray(), Out: ti.types.ndarray(), batch: int, c: int, h: int, w: int, stride: int):
    oh, ow = h // stride, w // stride
    for b, ch, i, j in ti.ndrange(batch, c, oh, ow):
        max_val = -1e30
        for dy, dx in ti.ndrange(stride, stride):
            idx_x = (((b * c + ch) * h + (i * stride + dy)) * w + (j * stride + dx))
            val = X[idx_x]
            if val > max_val: max_val = val
        Out[((b * c + ch) * oh + i) * ow + j] = max_val

@ti.kernel
def k_upsample2d_1d(X: ti.types.ndarray(), Out: ti.types.ndarray(), batch: int, c: int, h: int, w: int, scale: int):
    oh, ow = h * scale, w * scale
    for b, ch, i, j in ti.ndrange(batch, c, oh, ow):
        idx_x = (((b * c + ch) * h + (i // scale)) * w + (j // scale))
        Out[((b * c + ch) * oh + i) * ow + j] = X[idx_x]

@ti.kernel
def k_concat_1d(A: ti.types.ndarray(), B: ti.types.ndarray(), Out: ti.types.ndarray(), batch: int, ca: int, cb: int, h: int, w: int):
    co = ca + cb
    for b, i, j in ti.ndrange(batch, h, w):
        for c in range(ca):
            Out[((b * co + c) * h + i) * w + j] = A[((b * ca + c) * h + i) * w + j]
        for c in range(cb):
            Out[((b * co + (ca + c)) * h + i) * w + j] = B[((b * cb + c) * h + i) * w + j]

@ti.kernel
def k_add(A: ti.types.ndarray(), B: ti.types.ndarray(), Total: int):
    for i in range(Total):
        A[i] += B[i]

@ti.kernel
def k_mul(A: ti.types.ndarray(), B: ti.types.ndarray(), Total: int):
    for i in range(Total):
        # Result stored in A
        A[i] *= B[i]

@ti.kernel
def k_scale(A: ti.types.ndarray(), scale: float, Total: int):
    for i in range(Total):
        A[i] *= scale

@ti.kernel
def k_kv_cache_update(K_cache: ti.types.ndarray(), V_cache: ti.types.ndarray(),
                     K_new: ti.types.ndarray(), V_new: ti.types.ndarray(),
                     B: int, L_new: int, H: int, D: int, pos_offset: int, max_len: int):
    for b, h, i in ti.ndrange(B, H, L_new):
        for d in range(D):
            idx_new = (((b * L_new + i) * H + h) * D + d)
            idx_cache = (((b * max_len + (pos_offset + i)) * H + h) * D + d)
            K_cache[idx_cache] = K_new[idx_new]
            V_cache[idx_cache] = V_new[idx_new]

@ti.kernel
def k_attention(Q: ti.types.ndarray(), K: ti.types.ndarray(), V: ti.types.ndarray(), Out: ti.types.ndarray(),
                B: int, L_q: int, L_k: int, H: int, D: int, scale: float, pos_offset: int, window: int, softcap: float):
    # Causal Scaled Dot-Product Attention with Sliding Window support
    for b, h, i in ti.ndrange(B, H, L_q):
        i_abs = pos_offset + i
        causal_limit = i_abs + 1
        if causal_limit > L_k: causal_limit = L_k
        
        # 1. Calculate Max for Softmax stability
        max_score = -1e30
        for j in range(causal_limit):
            if window > 0 and (i_abs - j) >= window: continue
            
            score = 0.0
            for d in range(D):
                idx_q = (((b * L_q + i) * H + h) * D + d)
                idx_k = (((b * L_k + j) * H + h) * D + d)
                score += Q[idx_q] * K[idx_k]
            score *= scale
            if softcap > 0:
                score = ti.tanh(score / softcap) * softcap
            if score > max_score: max_score = score
            
        # 2. Calculate Softmax Denominator (ExpSum)
        expsum = 0.0
        for j in range(causal_limit):
            if window > 0 and (i_abs - j) >= window: continue
            score = 0.0
            for d in range(D):
                score += Q[(((b * L_q + i) * H + h) * D + d)] * K[(((b * L_k + j) * H + h) * D + d)]
            score *= scale
            if softcap > 0:
                score = ti.tanh(score / softcap) * softcap
            expsum += ti.exp(score - max_score)
            
        # 3. Weighted Sum of Values
        for j in range(causal_limit):
            if window > 0 and (i_abs - j) >= window: continue
            
            dot = 0.0
            for d in range(D):
                idx_q = (((b * L_q + i) * H + h) * D + d)
                idx_k = (((b * L_k + j) * H + h) * D + d)
                dot += Q[idx_q] * K[idx_k]
            dot *= scale
            if softcap > 0:
                dot = ti.tanh(dot / softcap) * softcap
            
            softmax_score = ti.exp(dot - max_score) / expsum

            for d3 in range(D):
                out_idx = (((b * L_q + i) * H + h) * D + d3)
                v_idx = (((b * L_k + j) * H + h) * D + d3)
                Out[out_idx] += softmax_score * V[v_idx]

class Tensor:
    """A wrapper around ti.ndarray for Vulkan NN operations."""
    def __init__(self, data, shape=None, dtype=ti.f32):
        if data is None:
            self.shape = shape
            total_size = 1
            for s in shape: total_size *= s
            self.arr = ti.ndarray(dtype=dtype, shape=(total_size,))
        elif isinstance(data, (np.ndarray, list)):
            np_arr = np.array(data)
            self.shape = np_arr.shape
            total_size = np_arr.size
            self.arr = ti.ndarray(dtype=dtype, shape=(total_size,))
            self.arr.from_numpy(np_arr.flatten().astype(np.float32 if dtype == ti.f32 else np.int32))
        elif isinstance(data, ti.Ndarray):
            # If already 1D but has extra shape metadata
            self.arr = data
            self.shape = data.shape if shape is None else shape
        elif isinstance(data, (float, int)):
            self.shape = (1,)
            self.arr = ti.ndarray(dtype=dtype, shape=(1,))
            self.arr.from_numpy(np.array([data], dtype=np.float32 if dtype == ti.f32 else np.int32))

    def to_numpy(self):
        # Result from ti.ndarray is 1D, reshape to logical shape
        return self.arr.to_numpy().reshape(self.shape)

    @property
    def total_size(self):
        sz = 1
        for s in self.shape: sz *= s
        return sz

    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, val):
        self._shape = val

    def clone(self):
        """Create a deep copy with its own GPU buffer (GPU-only, no CPU roundtrip)."""
        new_t = Tensor(None, shape=self.shape)
        k_copy(self.arr, new_t.arr, self.total_size)
        return new_t

    def squeeze(self, dim=None):
        if dim is None:
            new_shape = tuple(s for s in self.shape if s != 1)
        else:
            new_shape = list(self.shape)
            if new_shape[dim] == 1:
                new_shape.pop(dim)
            new_shape = tuple(new_shape)
        self.shape = new_shape
        return self

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        self.shape = tuple(shape)
        return self

    def view(self, *shape):
        return self.reshape(*shape)

    def save_to_disk(self, path):
        """Save tensor data to a binary file on disk."""
        data = self.to_numpy()
        data.tofile(path)
        print(f"Tensor saved to disk: {path} ({data.nbytes / 1024:.1f} KB)")

    @classmethod
    def from_disk(cls, path, shape, dtype=ti.f32):
        """Load tensor data from a binary file on disk to RAM (numpy)."""
        np_dtype = np.float32 if dtype == ti.f32 else np.int32
        data = np.fromfile(path, dtype=np_dtype).reshape(shape)
        return cls(data, dtype=dtype)

class Module:
    """Base class for all VulkanNN layers, mimicking torch.nn.Module."""
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def state_dict(self):
        sd = {}
        for name, param in self._parameters.items():
            sd[name] = param
        for name, module in self._modules.items():
            sub_sd = module.state_dict()
            for sub_name, sub_param in sub_sd.items():
                sd[f"{name}.{sub_name}"] = sub_param
        return sd

    def load_state_dict(self, state_dict):
        for name, param in self._parameters.items():
            if name in state_dict:
                param.arr.from_numpy(state_dict[name].astype(np.float32))
        for m_name, module in self._modules.items():
            child_sd = {k[len(m_name)+1:]: v for k, v in state_dict.items() if k.startswith(f"{m_name}.")}
            module.load_state_dict(child_sd)

    def to(self, device):
        # We assume Vulkan for now, this is a shim
        return self

class ModuleList(Module):
    """A list-like module wrapper for sequential/repeated layers."""
    def __init__(self, modules=None):
        super().__init__()
        self._layers = []
        if modules:
            for m in modules:
                self.append(m)

    def append(self, module):
        idx = len(self._layers)
        self._layers.append(module)
        self._modules[str(idx)] = module

    def __getitem__(self, idx):
        return self._layers[idx]

    def __len__(self):
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)

    def load_state_dict(self, state_dict):
        for name, param in self._parameters.items():
            if name in state_dict:
                param.arr.from_numpy(state_dict[name].astype(np.float32))
        for m_name, module in self._modules.items():
            child_sd = {k[len(m_name)+1:]: v for k, v in state_dict.items() if k.startswith(f"{m_name}.")}
            module.load_state_dict(child_sd)

    def to(self, device):
        # We assume Vulkan for now, this is a shim
        return self

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Use zeros instead of randn for faster initialization; weights are typically loaded from disk.
        w_data = np.zeros((in_features, out_features), dtype=np.float32)
        self.weight = Tensor(w_data)
        self.has_bias = bias
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        # Linear in torch works on the last dimension
        N_out = self.out_features
        K = self.in_features
        
        # Flatten leading dimensions
        M = 1
        for s in orig_shape[:-1]: M *= s
        
        out = Tensor(None, shape=(M, N_out))
        k_matmul(x.arr, self.weight.arr, out.arr, M, N_out, K)
        if self.has_bias:
            k_add_bias(out.arr, self.bias.arr, M, N_out)
        
        # Reshape result back to (..., N_out)
        new_shape = list(orig_shape[:-1]) + [N_out]
        out.shape = tuple(new_shape)
        return out

class TiledLinear(Module):
    """A Linear layer that pages weights from RAM to VRAM in tiles."""
    def __init__(self, in_features, out_features, bias=True, tile_size=1024):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        
        # Store weights in RAM (numpy). Use zeros for fast initialization.
        self.weight_ram = np.zeros((in_features, out_features), dtype=np.float32)
        
        self.has_bias = bias
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32))
            
        # Allocate a REUSABLE VRAM tile buffer
        self.weight_tile_vram = ti.ndarray(dtype=ti.f32, shape=(in_features * tile_size,))

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        N_out = self.out_features
        K = self.in_features
        M = 1
        for s in orig_shape[:-1]: M *= s
        
        # Initialize output with zero (partial sum accumulation)
        out = Tensor(None, shape=(M, N_out))
        k_zero(out.arr, out.total_size)
        
        # Stream tiles from RAM to VRAM
        for n_offset in range(0, N_out, self.tile_size):
            n_tile = min(self.tile_size, N_out - n_offset)
            
            # Slice RAM and Upload to VRAM tile buffer (with padding if needed)
            full_tile = np.zeros((K, self.tile_size), dtype=np.float32)
            full_tile[:, :n_tile] = self.weight_ram[:, n_offset : n_offset + n_tile]
            self.weight_tile_vram.from_numpy(full_tile.flatten())
            
            # Compute partial result on GPU using stride
            k_matmul_tiled(x.arr, self.weight_tile_vram, out.arr, M, N_out, K, n_offset, n_tile, self.tile_size)
            
        if self.has_bias:
            k_add_bias(out.arr, self.bias.arr, M, N_out)
            
        new_shape = list(orig_shape[:-1]) + [N_out]
        out.shape = tuple(new_shape)
        return out

class MatFormerLinear(TiledLinear):
    """A TiledLinear layer that supports 'Matryoshka' slicing."""
    def forward(self, x: Tensor, sub_out_features=None) -> Tensor:
        if sub_out_features is None:
            return super().forward(x)
        
        # Dynamic slicing: only use the first N columns of the weight matrix
        N_out = sub_out_features
        orig_shape = x.shape
        K = self.in_features
        M = 1
        for s in orig_shape[:-1]: M *= s
        
        out = Tensor(None, shape=(M, N_out))
        k_zero(out.arr, out.total_size)
        
        for n_offset in range(0, N_out, self.tile_size):
            n_tile = min(self.tile_size, N_out - n_offset)
            full_tile = np.zeros((K, self.tile_size), dtype=np.float32)
            full_tile[:, :n_tile] = self.weight_ram[:, n_offset : n_offset + n_tile]
            self.weight_tile_vram.from_numpy(full_tile.flatten())
            k_matmul_tiled(x.arr, self.weight_tile_vram, out.arr, M, N_out, K, n_offset, n_tile, self.tile_size)
            
        if self.has_bias:
            # Masked bias or sliced bias? Gemma usually has sliced bias.
            k_add_bias(out.arr, self.bias.arr, M, N_out) # k_add_bias indices from bias[j] so it works if bias is 1D
            
        new_shape = list(orig_shape[:-1]) + [N_out]
        out.shape = tuple(new_shape)
        return out

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        N = shape[-1]
        M = x.total_size // N
        k_softmax_1d(x.arr, M, N)
        return x

class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = Tensor(np.ones(dim, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        N = shape[-1]
        M = x.total_size // N
        k_rmsnorm_1d(x.arr, self.weight.arr, M, N, self.eps)
        return x

class GELUTanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        k_gelu_tanh(x.arr, x.total_size)
        return x

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        w_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        self.weight = Tensor(w_data)

    def forward(self, x: Tensor) -> Tensor:
        B, L = x.shape
        D = self.weight.shape[1]
        out = Tensor(None, shape=(B, L, D))
        k_embedding_1d(x.arr, self.weight.arr, out.arr, B, L, D)
        return out

class TiledEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Keep weights in RAM. Use zeros for fast initialization.
        self.weight_ram = np.zeros((num_embeddings, embedding_dim), dtype=np.float32)
        # Small temporary buffer for transfer if needed, but for embedding we can direct copy
        self.weight = Tensor(None, shape=(1, embedding_dim)) # Placeholder for state_dict compatibility

    def forward(self, x: Tensor) -> Tensor:
        # x is (B, L)
        indices = x.to_numpy().flatten().astype(np.int32)
        B, L = x.shape
        D = self.embedding_dim
        
        # Fetch only the required rows from RAM
        # This is very memory efficient for sparse lookups (LLM inference)
        vectors = self.weight_ram[indices].reshape(B, L, D)
        
        # Create output tensor on GPU and copy data
        out = Tensor(vectors)
        return out

    def load_weight(self, data):
        self.weight_ram = data
        self.weight.shape = data.shape # Update placeholder shape

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        else:
            self.kh, self.kw = kernel_size
        w_data = np.random.randn(out_channels, in_channels, self.kh, self.kw).astype(np.float32) * 0.1
        self.weight = Tensor(w_data)
        self.has_bias = bias
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        b, ic, h, w = x.shape
        oh, ow = h - self.kh + 1, w - self.kw + 1
        out = Tensor(None, shape=(b, self.out_channels, oh, ow))
        k_conv2d_1d(x.arr, self.weight.arr, self.bias.arr, out.arr, b, ic, self.out_channels, h, w, self.kh, self.kw)
        return out

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        k_relu_1d(x.arr, x.total_size)
        return x

class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
    def forward(self, x: Tensor) -> Tensor:
        k_leaky_relu_1d(x.arr, x.total_size, self.alpha)
        return x

class GaussianTopK(Module):
    def __init__(self, sparsity=0.0):
        super().__init__()
        self.sparsity = sparsity
        # Precompute std_multiplier (Normal distribution icdf)
        # For simplicity, we use a lookup or common value since we only have 0.95 or 0.0
        if sparsity >= 0.95:
            self.std_multiplier = 1.64485362695  # icdf(0.95)
        elif sparsity > 0.0:
            import scipy.stats
            self.std_multiplier = float(scipy.stats.norm.ppf(sparsity))
        else:
            self.std_multiplier = -1e10 # Effectively no cutoff
            
    def forward(self, x: Tensor) -> Tensor:
        if self.sparsity > 0.0:
            shape = x.shape
            N = shape[-1]
            k_gaussian_topk(x.arr, x.total_size, N, self.std_multiplier)
        return x

class Gemma3nLaurelBlock(Module):
    def __init__(self, hidden_size, laurel_rank, eps=1e-6):
        super().__init__()
        self.linear_left = Linear(hidden_size, laurel_rank, bias=False)
        self.linear_right = Linear(laurel_rank, hidden_size, bias=False)
        self.post_laurel_norm = RMSNorm(hidden_size, eps=eps)
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        # Bottleneck residual
        res = self.linear_left(hidden_states)
        res = self.linear_right(res)
        res = self.post_laurel_norm(res)
        return res # We return just the part to be added

class Gemma3nAltUp(Module):
    def __init__(self, hidden_size, num_inputs, active_idx=0, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.active_idx = active_idx
        
        self.modality_router = Linear(hidden_size, num_inputs, bias=False)
        self.router_norm = RMSNorm(hidden_size, eps=eps)
        self.router_input_scale = hidden_size**-1.0
        
        self.prediction_coefs = Linear(num_inputs, num_inputs**2, bias=False)
        self.correction_coefs = Linear(num_inputs, num_inputs, bias=False)
        
        self.correct_output_scale = Tensor(np.zeros(hidden_size, dtype=np.float32))

    def compute_router_modalities(self, x: Tensor) -> Tensor:
        # We need a copy of x as RMSNorm is in-place
        router_inputs = x.clone()
        self.router_norm(router_inputs)
        k_scale(router_inputs.arr, self.router_input_scale, router_inputs.total_size)
        routed = self.modality_router(router_inputs)
        # Apply tanh in-place on routed
        for i in range(routed.total_size):
            routed.arr[i] = ti.tanh(routed.arr[i])
        return routed

    def predict(self, hidden_states: Tensor) -> Tensor:
        # hidden_states is 4D [S, B, L, D]
        # 1. Get modality for active slice
        B, L, D = hidden_states.shape[1], hidden_states.shape[2], hidden_states.shape[3]
        active_slice = Tensor(None, shape=(B, L, D))
        # Logic to extract active_slice from 4D tensor... 
        # For efficiency, we can pass the whole 4D to a kernel
        
        # Actually, let's just use the router on the active slice data
        modalities = self.compute_router_modalities(Tensor(hidden_states.arr, shape=(B, L, D), offset=self.active_idx * B * L * D))
        
        all_coefs = self.prediction_coefs(modalities) # [B, L, S*S]
        
        out = Tensor(None, shape=hidden_states.shape)
        k_altup_predict(hidden_states.arr, all_coefs.arr, out.arr, self.num_inputs, B, L, D)
        return out

    def correct(self, predictions: Tensor, activated: Tensor) -> Tensor:
        # predictions is [S, B, L, D], activated is [B, L, D]
        modalities = self.compute_router_modalities(activated)
        all_coefs = self.correction_coefs(modalities) # [B, L, S]
        # In modeling_gemma3n.py: all_coefs = correction_coefs(modalities) + 1.0
        for i in range(all_coefs.total_size):
            all_coefs.arr[i] += 1.0
            
        k_altup_correct(predictions.arr, activated.arr, all_coefs.arr, self.num_inputs, activated.shape[0], activated.shape[1], activated.shape[2], self.active_idx)
        return predictions

    def scale_corrected_output(self, corrected: Tensor) -> Tensor:
        # Apply correct_output_scale (per-dim scale)
        # corrected is [B, L, D]
        for b, l, d in ti.ndrange(corrected.shape[0], corrected.shape[1], corrected.shape[2]):
            idx = (b * corrected.shape[1] + l) * corrected.shape[2] + d
            corrected.arr[idx] *= self.correct_output_scale.arr[d]
        return corrected

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        out = Tensor(None, shape=(b, c, h // self.stride, w // self.stride))
        k_pool2d_1d(x.arr, out.arr, b, c, h, w, self.stride)
        return out

class Upsample(Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        out = Tensor(None, shape=(b, c, h * self.scale_factor, w * self.scale_factor))
        k_upsample2d_1d(x.arr, out.arr, b, c, h, w, self.scale_factor)
        return out

class Concatenate(Module):
    def forward(self, tensors) -> Tensor:
        # Mimic torch.cat((a, b), dim=1)
        a, b = tensors
        ba, ca, ha, wa = a.shape
        bb, cb, hb, wb = b.shape
        out = Tensor(None, shape=(ba, ca + cb, ha, wa))
        k_concat_1d(a.arr, b.arr, out.arr, ba, ca, cb, ha, wa)
        return out

class RoPE(Module):
    """Rotary Positional Embedding layer."""
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, pos_offset: int = 0) -> Tensor:
        B, L, H, D = x.shape
        k_rope(x.arr, cos.arr, sin.arr, B, L, H, D, pos_offset)
        return x

class Gemma3nPLE(Module):
    def __init__(self, hidden_size, ple_dim, eps=1e-6):
        super().__init__()
        self.per_layer_input_gate = Linear(hidden_size, ple_dim, bias=False)
        self.per_layer_projection = Linear(ple_dim, hidden_size, bias=False)
        self.post_per_layer_input_norm = RMSNorm(hidden_size, eps=eps)
        self.activation = GELUTanh() # Standard for Gemma 3n
        
    def forward(self, active_state: Tensor, per_layer_input: Tensor, full_state: Tensor):
        # active_state: [B, L, D], per_layer_input: [B, L, ple_dim], full_state: [S, B, L, D]
        # 1. Gate calculation
        gate = self.per_layer_input_gate(active_state)
        self.activation(gate)
        # 2. Add PLE info
        k_mul(gate.arr, per_layer_input.arr, gate.total_size)
        # 3. Projection
        proj = self.per_layer_projection(gate)
        self.post_per_layer_input_norm(proj)
        # 4. Add to sparse slices (usually 1:end)
        S, B, L, D = full_state.shape
        k_add_to_slices(proj.arr, full_state.arr, 1, S, B, L, D)

class Gemma3Block(Module):
    """A real Gemma 3 Transformer Block with AltUP, Laurel, and PLE support."""
    def __init__(self, hidden_size=2048, num_heads=8, num_kv_heads=2, head_dim=256, 
                 intermediate_size=8192, tile_size=1024, layer_type="full_attention", window=512,
                 altup_num_inputs=4, laurel_rank=64, ple_dim=256, sparsity=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads
        self.window = window if layer_type == "sliding_attention" else 0
        self.active_idx = 0 # Default for AltUP
        self.altup_correct_scale = True # Enabled for 3n

        # Specialized 3n components
        self.altup = Gemma3nAltUp(hidden_size, altup_num_inputs, active_idx=self.active_idx)
        self.laurel = Gemma3nLaurelBlock(hidden_size, laurel_rank)
        self.ple = Gemma3nPLE(hidden_size, ple_dim)
        self.sparsity_gate = GaussianTopK(sparsity)

        # Attention Projections
        self.q_proj = MatFormerLinear(hidden_size, num_heads * head_dim, bias=False, tile_size=tile_size)
        self.k_proj = MatFormerLinear(hidden_size, num_kv_heads * head_dim, bias=False, tile_size=tile_size)
        self.v_proj = MatFormerLinear(hidden_size, num_kv_heads * head_dim, bias=False, tile_size=tile_size)
        self.o_proj = MatFormerLinear(num_heads * head_dim, hidden_size, bias=False, tile_size=tile_size)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # FFN
        self.gate_proj = MatFormerLinear(hidden_size, intermediate_size, bias=False, tile_size=tile_size)
        self.up_proj   = MatFormerLinear(hidden_size, intermediate_size, bias=False, tile_size=tile_size)
        self.down_proj = MatFormerLinear(intermediate_size, hidden_size, bias=False, tile_size=tile_size)
        
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size)
        self.post_feedforward_layernorm = RMSNorm(hidden_size)

        self.rope = RoPE()
        self.activation = GELUTanh()

    def forward(self, hidden_states: Tensor, cos: Tensor, sin: Tensor, 
                ple_input: Tensor, k_cache: Tensor = None, v_cache: Tensor = None, 
                pos_offset: int = 0, active_intermediate_size=None) -> Tensor:
        # hidden_states is 4D [S, B, L, D]
        # 1. AltUP Predict
        predictions = self.altup.predict(hidden_states)
        
        # 2. Extract Active Slice
        S, B, L_new, D_hidden = predictions.shape
        active_prediction = Tensor(None, shape=(B, L_new, D_hidden))
        k_extract_slice(predictions.arr, active_prediction.arr, self.active_idx, B, L_new, D_hidden)
        
        # 3. Attention Path
        active_prediction_normed = self.input_layernorm(active_prediction.clone())
        laurel_output = self.laurel(active_prediction_normed) # Bottle-neck residual part
        
        q = self.q_proj(active_prediction_normed)
        k = self.k_proj(active_prediction_normed)
        v = self.v_proj(active_prediction_normed)
        
        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim
        
        q.reshape(B, L_new, H_q, D)
        k.reshape(B, L_new, H_kv, D)
        v.reshape(B, L_new, H_kv, D)

        self.q_norm(q)
        self.k_norm(k)
        
        self.rope(q, cos, sin, pos_offset)
        self.rope(k, cos, sin, pos_offset)
        
        if H_kv < H_q:
            k_exp = Tensor(None, shape=(B, L_new, H_q, D))
            v_exp = Tensor(None, shape=(B, L_new, H_q, D))
            k_repeat_kv(k.arr, k_exp.arr, B, L_new, H_kv, D, self.num_groups)
            k_repeat_kv(v.arr, v_exp.arr, B, L_new, H_kv, D, self.num_groups)
            k, v = k_exp, v_exp
        
        L_total = L_new
        if k_cache is not None:
             max_len = k_cache.shape[1]
             k_kv_cache_update(k_cache.arr, v_cache.arr, k.arr, v.arr, B, L_new, H_q, D, pos_offset, max_len)
             k_full = k_cache
             v_full = v_cache
             L_total = pos_offset + L_new
        else:
             k_full = k
             v_full = v

        attn_out_res = Tensor(None, shape=(B, L_new, H_q, D))
        k_zero(attn_out_res.arr, attn_out_res.total_size)
        attn_scale = 1.0 / ti.sqrt(float(D))
        attn_softcap = 50.0 # From gemma3n_introduction.md
        k_attention(q.arr, k_full.arr, v_full.arr, attn_out_res.arr, B, L_new, L_total, H_q, D, attn_scale, pos_offset, self.window, attn_softcap)
        
        attn_out = attn_out_res.reshape(B, L_new, H_q * D)
        attn_out = self.o_proj(attn_out)
        attn_out = self.post_attention_layernorm(attn_out)
        
        # 4. Attention Residual & Laurel Gating
        # attn_gated = active_prediction + attn_out
        k_add(active_prediction.arr, attn_out.arr, active_prediction.total_size)
        # attn_laurel = (attn_gated + laurel_output) / sqrt(2)
        k_add(active_prediction.arr, laurel_output.arr, active_prediction.total_size)
        k_scale(active_prediction.arr, 0.70710678, active_prediction.total_size) # 1/sqrt(2)
        
        # 5. FFN Path
        ffn_in = self.pre_feedforward_layernorm(active_prediction.clone())
        
        # MatFormer Slicing for intermediate size (Selective activation per request)
        i_size = active_intermediate_size if active_intermediate_size else self.intermediate_size
        gate = self.gate_proj(ffn_in, sub_out_features=i_size)
        up = self.up_proj(ffn_in, sub_out_features=i_size)
        
        self.sparsity_gate(gate)
        self.activation(gate)
        k_mul(gate.arr, up.arr, gate.total_size)
        
        ffn_out = self.down_proj(gate) # Automatically handles slicing internally
        ffn_out = self.post_feedforward_layernorm(ffn_out)
        
        # 6. FFN Residual
        # ffw_laurel_gated = attn_laurel + ffw_out
        k_add(active_prediction.arr, ffn_out.arr, active_prediction.total_size)
        
        # 7. AltUP Correct
        self.altup.correct(predictions, active_prediction)
        
        # 8. Correction Scaling (Source L1396-L1398)
        if self.altup_correct_scale:
            self.altup.scale_corrected_output(active_prediction)

        # 9. PLE Gating & Accumulation
        self.ple(active_prediction, ple_input, predictions)
        
        return predictions

class Gemma3Model(Module):
    """A real Gemma 3 Transformer Model with 4D AltUP state management."""
    def __init__(self, num_layers=30, hidden_size=2048, num_heads=8, num_kv_heads=2, 
                 head_dim=256, intermediate_size=8192, vocab_size=262400, tile_size=1024,
                 layer_types=None, sliding_window=512,
                 altup_num_inputs=4, laurel_rank=64, ple_dim=256, sparsity_pattern=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.altup_num_inputs = altup_num_inputs
        self.active_idx = 0
        
        self.embed_tokens = TiledEmbedding(vocab_size, hidden_size)
        # 3n specialized: projections to expand 2D embeddings to 4D stack
        self.altup_projections = ModuleList()
        for i in range(1, altup_num_inputs):
            self.altup_projections.append(Linear(hidden_size, hidden_size, bias=False))
            
        # 3n specialized: projections to collapse 4D stack back to 2D
        self.altup_unembed_projections = ModuleList()
        for i in range(1, altup_num_inputs):
            self.altup_unembed_projections.append(Linear(hidden_size, hidden_size, bias=False))

        self.layers = ModuleList()
        print("Initializing Transformer layers...")
        for i in range(num_layers):
            if i % 5 == 0: print(f"Building layer {i}/{num_layers}...")
            l_type = layer_types[i] if layer_types else "full_attention"
            sparsity = sparsity_pattern[i] if sparsity_pattern else 0.0
            self.layers.append(Gemma3Block(
                hidden_size=hidden_size, 
                num_heads=num_heads, 
                num_kv_heads=num_kv_heads, 
                head_dim=head_dim, 
                intermediate_size=intermediate_size,
                tile_size=tile_size,
                layer_type=l_type,
                window=sliding_window,
                altup_num_inputs=altup_num_inputs,
                laurel_rank=laurel_rank,
                ple_dim=ple_dim,
                sparsity=sparsity
            ))
        
        self.norm = RMSNorm(hidden_size)
        print("Initializing language modeling head...")
        self.lm_head = MatFormerLinear(hidden_size, vocab_size, bias=False, tile_size=tile_size)

    def forward(self, input_ids: Tensor, cos: Tensor, sin: Tensor, 
                ple_inputs: Tensor, caches=None, pos_offset=0, active_intermediate_size=None) -> Tensor:
        # 1. Base Embedding
        x0 = self.embed_tokens(input_ids)
        embed_scale = float(np.sqrt(self.hidden_size))
        k_scale(x0.arr, embed_scale, x0.total_size)
        
        # 2. Initialization of 4D AltUP State
        B, L, D = x0.shape
        S = self.altup_num_inputs
        h_state = Tensor(None, shape=(S, B, L, D))
        
        # Copy active slice directly
        k_copy(x0.arr, Tensor(h_state.arr, shape=(B, L, D), offset=self.active_idx * B * L * D).arr, x0.total_size)
        
        # Project other slices and apply magnitude norm
        for i in range(1, S):
            proj_out = self.altup_projections[i-1](x0)
            target_slice = Tensor(h_state.arr, shape=(B, L, D), offset=i * B * L * D)
            k_copy(proj_out.arr, target_slice.arr, proj_out.total_size)
            # Magnitude Norm (Source L1706-L1708)
            k_magnitude_norm(h_state.arr, h_state.arr, i, self.active_idx, B, L, D, 1e-5)
            
        # 3. Transformer Layers Loop
        for i, layer in enumerate(self.layers):
            k_cache = caches[i][0] if caches else None
            v_cache = caches[i][1] if caches else None
            # ple_inputs is [B, L, num_layers, PLE_DIM]
            layer_ple = Tensor(ple_inputs.arr, shape=(B, L, ple_inputs.shape[3]), offset=i * B * L * ple_inputs.shape[3])
            h_state = layer(h_state, cos, sin, layer_ple, k_cache, v_cache, pos_offset, 
                            active_intermediate_size=active_intermediate_size)
            
        # 4. Collapse 4D State to 2D
        # unembed projections and average
        temp_sum = h_state.clone() # We'll reuse this to sum
        # Slice 0 stays same
        for i in range(1, S):
            slice_i = Tensor(h_state.arr, shape=(B, L, D), offset=i * B * L * D)
            unemb_proj = self.altup_unembed_projections[i-1](slice_i)
            # Magnitude Norm before combining
            target_slice = Tensor(temp_sum.arr, shape=(B, L, D), offset=0) # Index 0 is target
            dest_slice = Tensor(temp_sum.arr, shape=(B, L, D), offset=i * B * L * D)
            k_copy(unemb_proj.arr, dest_slice.arr, unemb_proj.total_size)
            k_magnitude_norm(temp_sum.arr, temp_sum.arr, i, 0, B, L, D, 1e-5)
            
        # Average all slices into [B, L, D]
        x_out = Tensor(None, shape=(B, L, D))
        k_zero(x_out.arr, x_out.total_size)
        for i in range(S):
            slice_i = Tensor(temp_sum.arr, shape=(B, L, D), offset=i * B * L * D)
            k_add(x_out.arr, slice_i.arr, x_out.total_size)
        k_scale(x_out.arr, 1.0/S, x_out.total_size)
        
        # 5. Output Norm and Heads
        x_out = self.norm(x_out)
        logits = self.lm_head(x_out)
        return logits

class Gemma3ForMultimodalLM(Module):
    """Official wrapper structure from PyTorch Gemma 3 Implementation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = Gemma3Model(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size[0] if isinstance(config.intermediate_size, list) else config.intermediate_size,
            vocab_size=config.vocab_size,
            layer_types=config.layer_types,
            sliding_window=config.sliding_window,
            altup_num_inputs=config.altup_num_inputs,
            laurel_rank=config.laurel_rank,
            ple_dim=config.hidden_size_per_layer_input
        )
        # Placeholders for multimodal encoders (Conditional loading support)
        self.vision_encoder = None
        self.audio_encoder = None

    def forward(self, input_ids: Tensor, cos: Tensor, sin: Tensor, 
                ple_inputs: Tensor, caches=None, pos_offset=0) -> Tensor:
        # Text only for now, matching conditional loading logic
        return self.language_model(input_ids, cos, sin, ple_inputs, caches, pos_offset)

def get_cos_sin(seq_len, head_dim, base=10000):
    """Generate RoPE cos/sin cache."""
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs), np.sin(freqs)

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

# Submodules for easier importing
class nn:
    Module = Module
    Linear = Linear
    TiledLinear = TiledLinear
    MatFormerLinear = MatFormerLinear
    Conv2d = Conv2d
    ReLU = ReLU
    LeakyReLU = LeakyReLU
    GELUTanh = GELUTanh
    Softmax = Softmax
    RMSNorm = RMSNorm
    Embedding = Embedding
    TiledEmbedding = TiledEmbedding
    MaxPool2d = MaxPool2d
    Upsample = Upsample
    Sequential = Sequential

class F:
    @staticmethod
    def relu(x): return ReLU()(x)
    @staticmethod
    def leaky_relu(x, alpha=0.01): return LeakyReLU(alpha)(x)
    @staticmethod
    def max_pool2d(x, kernel_size=2): return MaxPool2d(kernel_size)(x)
    @staticmethod
    def silu(x): return SiLU()(x)
    @staticmethod
    def softmax(x, dim=-1): return Softmax(dim)(x)

# Testing and verification
if __name__ == "__main__":
    print("VulkanNN Module Loaded Directly")
