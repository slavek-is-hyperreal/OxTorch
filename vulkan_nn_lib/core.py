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
                B: int, L_q: int, L_k: int, H: int, D: int, scale: float, pos_offset: int, window: int):
    # Causal Scaled Dot-Product Attention with Sliding Window support
    for b, h, i in ti.ndrange(B, H, L_q):
        # i is current token index relative to start of Q
        i_abs = pos_offset + i
        
        # Causal: query at position i_abs can attend to keys at positions 0..i_abs
        causal_limit = i_abs + 1
        if causal_limit > L_k:
            causal_limit = L_k
        
        # 1. Calculate Max for Softmax stability
        max_score = -1e30
        for j in range(causal_limit):
            # Sliding Window Mask: i_abs - j must be < window (window <= 0 means full attention)
            if window > 0 and (i_abs - j) >= window: continue
            
            attn_score = 0.0
            for d in range(D):
                idx_q = (((b * L_q + i) * H + h) * D + d)
                idx_k = (((b * L_k + j) * H + h) * D + d)
                attn_score += Q[idx_q] * K[idx_k]
            attn_score *= scale
            if attn_score > max_score: max_score = attn_score
            
        # 2. Calculate Softmax Denominator (ExpSum)
        expsum = 0.0
        for j in range(causal_limit):
            if window > 0 and (i_abs - j) >= window: continue
            
            score = 0.0
            for d in range(D):
                score += Q[(((b * L_q + i) * H + h) * D + d)] * K[(((b * L_k + j) * H + h) * D + d)]
            expsum += ti.exp(score * scale - max_score)
            
        # 3. Weighted Sum of Values
        for j in range(causal_limit):
            if window > 0 and (i_abs - j) >= window: continue
            
            score = 0.0
            for d2 in range(D):
                score += Q[(((b * L_q + i) * H + h) * D + d2)] * K[(((b * L_k + j) * H + h) * D + d2)]
            
            softmax_score = ti.exp(score * scale - max_score) / expsum

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
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        k_leaky_relu_1d(x.arr, x.total_size, self.negative_slope)
        return x

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

class Gemma3Block(Module):
    """A real Gemma 3 Transformer Block with GQA and Sliding Window support."""
    def __init__(self, hidden_size=2048, num_heads=8, num_kv_heads=2, head_dim=256, 
                 intermediate_size=8192, tile_size=1024, layer_type="full_attention", window=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads  # GQA group ratio
        self.window = window if layer_type == "sliding_attention" else 0

        # Attention Projections (GQA: K/V are smaller)
        self.q_proj = MatFormerLinear(hidden_size, num_heads * head_dim, bias=False, tile_size=tile_size)
        self.k_proj = MatFormerLinear(hidden_size, num_kv_heads * head_dim, bias=False, tile_size=tile_size)
        self.v_proj = MatFormerLinear(hidden_size, num_kv_heads * head_dim, bias=False, tile_size=tile_size)
        self.o_proj = MatFormerLinear(num_heads * head_dim, hidden_size, bias=False, tile_size=tile_size)

        # MatFormer FFN (Gated MLP)
        self.gate_proj = MatFormerLinear(hidden_size, intermediate_size, bias=False, tile_size=tile_size)
        self.up_proj   = MatFormerLinear(hidden_size, intermediate_size, bias=False, tile_size=tile_size)
        self.down_proj = MatFormerLinear(intermediate_size, hidden_size, bias=False, tile_size=tile_size)
        
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.rope = RoPE()
        self.activation = GELUTanh()

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, 
                k_cache: Tensor = None, v_cache: Tensor = None, 
                pos_offset: int = 0, sub_features=None) -> Tensor:
        # Pre-LN Attention
        residual = x.clone()  # Deep copy: RMSNorm modifies x in-place
        x = self.input_layernorm(x)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        B, L_new, _ = q.shape
        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim
        
        q.reshape(B, L_new, H_q, D)
        k.reshape(B, L_new, H_kv, D)
        v.reshape(B, L_new, H_kv, D)
        
        self.rope(q, cos, sin, pos_offset)
        self.rope(k, cos, sin, pos_offset)
        
        # GQA: Repeat KV heads to match Q heads (GPU Optimized)
        if H_kv < H_q:
            k_exp = Tensor(None, shape=(B, L_new, H_q, D))
            v_exp = Tensor(None, shape=(B, L_new, H_q, D))
            k_repeat_kv(k.arr, k_exp.arr, B, L_new, H_kv, D, self.num_groups)
            k_repeat_kv(v.arr, v_exp.arr, B, L_new, H_kv, D, self.num_groups)
            k, v = k_exp, v_exp
        
        # KV-Cache Update
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

        # Real Scaled Dot-Product Attention
        attn_out = Tensor(None, shape=(B, L_new, H_q, D))
        k_zero(attn_out.arr, attn_out.total_size)  # Critical: zero before += accumulation
        scale = 1.0 / ti.sqrt(float(D))
        k_attention(q.arr, k_full.arr, v_full.arr, attn_out.arr, B, L_new, L_total, H_q, D, scale, pos_offset, self.window)
        
        x = attn_out.reshape(B, L_new, H_q * D)
        x = self.o_proj(x)
        
        k_add(x.arr, residual.arr, x.total_size)
        
        # Pre-LN FFN
        residual = x.clone()  # Deep copy: RMSNorm modifies x in-place
        x = self.post_attention_layernorm(x)
        
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        self.activation(gate)
        k_mul(gate.arr, up.arr, gate.total_size)
        
        x = self.down_proj(gate)
        k_add(x.arr, residual.arr, x.total_size)
        
        return x

class Gemma3Model(Module):
    """A real Gemma 3 Transformer Model."""
    def __init__(self, num_layers=30, hidden_size=2048, num_heads=8, num_kv_heads=2, 
                 head_dim=256, intermediate_size=8192, vocab_size=262400, tile_size=1024,
                 layer_types=None, sliding_window=512):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.embed_tokens = TiledEmbedding(vocab_size, hidden_size)
        
        self.layers = ModuleList()
        print("Initializing Transformer layers...")
        for i in range(num_layers):
            if i % 5 == 0: print(f"Building layer {i}/{num_layers}...")
            l_type = layer_types[i] if layer_types else "full_attention"
            self.layers.append(Gemma3Block(
                hidden_size=hidden_size, 
                num_heads=num_heads, 
                num_kv_heads=num_kv_heads, 
                head_dim=head_dim, 
                intermediate_size=intermediate_size,
                tile_size=tile_size,
                layer_type=l_type,
                window=sliding_window
            ))
        
        self.norm = RMSNorm(hidden_size)
        print("Initializing language modeling head...")
        self.lm_head = MatFormerLinear(hidden_size, vocab_size, bias=False, tile_size=tile_size)

    def forward(self, input_ids: Tensor, cos: Tensor, sin: Tensor, 
                caches=None, pos_offset=0) -> Tensor:
        x = self.embed_tokens(input_ids)
        
        # Gemma-specific: scale embeddings by sqrt(hidden_size)
        embed_scale = float(np.sqrt(self.hidden_size))
        k_scale(x.arr, embed_scale, x.total_size)
        
        for i, layer in enumerate(self.layers):
            k_cache = caches[i][0] if caches else None
            v_cache = caches[i][1] if caches else None
            x = layer(x, cos, sin, k_cache, v_cache, pos_offset)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

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
