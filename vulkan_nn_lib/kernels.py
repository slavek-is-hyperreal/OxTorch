import taichi as ti
import numpy as np
import os

# Initialize Taichi
ti.init(
    arch=ti.vulkan, 
    offline_cache=True,
    device_memory_GB=1.0 
)

@ti.kernel
def k_zero(X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        X[i] = 0.0

@ti.kernel
def k_add(A: ti.types.ndarray(), B: ti.types.ndarray(), size: int, size_b: int):
    for i in range(size):
        A[i] += B[i % size_b]

@ti.kernel
def k_sgd(p: ti.types.ndarray(), g: ti.types.ndarray(), lr: float, size: int):
    for i in range(size):
        p[i] -= lr * g[i]

k_sgd_step = k_sgd

@ti.kernel
def k_adam(p: ti.types.ndarray(), g: ti.types.ndarray(), m: ti.types.ndarray(), v: ti.types.ndarray(),
                lr: float, b1: float, b2: float, eps: float, t: int, size: int):
    b1_corr = 1.0 - pow(b1, float(t))
    b2_corr = 1.0 - pow(b2, float(t))
    
    for i in range(size):
        m[i] = b1 * m[i] + (1.0 - b1) * g[i]
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i]
        m_hat = m[i] / b1_corr
        v_hat = v[i] / b2_corr
        p[i] -= lr * m_hat / (ti.sqrt(v_hat) + eps)

k_adam_step = k_adam
k_adam_hybrid = k_adam

@ti.kernel
def k_sub(A: ti.types.ndarray(), B: ti.types.ndarray(), Total: int, B_Total: int):
    for i in range(Total):
        A[i] -= B[i % B_Total]

@ti.kernel
def k_mul(A: ti.types.ndarray(), B: ti.types.ndarray(), Total: int, B_Total: int):
    for i in range(Total):
        A[i] *= B[i % B_Total]

@ti.kernel
def k_div(A: ti.types.ndarray(), B: ti.types.ndarray(), Total: int, B_Total: int):
    for i in range(Total):
        A[i] /= B[i % B_Total]

@ti.kernel
def k_scale(X: ti.types.ndarray(), Scale: float, Total: int):
    for i in range(Total):
        X[i] *= Scale

@ti.kernel
def k_copy(Src: ti.types.ndarray(), Dst: ti.types.ndarray(), Total: int):
    for i in range(Total):
        Dst[i] = Src[i]

@ti.kernel
def k_add_backward(Grad_Out: ti.types.ndarray(), Grad_A: ti.types.ndarray(), Grad_B: ti.types.ndarray(), Total: int, B_Total: int):
    for i in range(Total):
        g = Grad_Out[i]
        ti.atomic_add(Grad_A[i], g)
        ti.atomic_add(Grad_B[i % B_Total], g)

@ti.kernel
def k_sub_backward(Grad_Out: ti.types.ndarray(), Grad_A: ti.types.ndarray(), Grad_B: ti.types.ndarray(), Total: int, B_Total: int):
    for i in range(Total):
        g = Grad_Out[i]
        Grad_A[i] += g
        ti.atomic_add(Grad_B[i % B_Total], -g)

@ti.kernel
def k_mul_backward(Grad_Out: ti.types.ndarray(), A: ti.types.ndarray(), B: ti.types.ndarray(), 
                   Grad_A: ti.types.ndarray(), Grad_B: ti.types.ndarray(), Total: int, B_Total: int):
    for i in range(Total):
        g = Grad_Out[i]
        Grad_A[i] += g * B[i % B_Total]
        ti.atomic_add(Grad_B[i % B_Total], g * A[i])

@ti.kernel
def k_scale_backward(Grad_Out: ti.types.ndarray(), Scale: float, Grad_X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        Grad_X[i] += Grad_Out[i] * Scale

@ti.kernel
def k_matmul_backward(Grad_Out: ti.types.ndarray(), A: ti.types.ndarray(), B: ti.types.ndarray(),
                      Grad_A: ti.types.ndarray(), Grad_B: ti.types.ndarray(), 
                      M: int, N: int, K: int):
    for i, k in ti.ndrange(M, K):
        acc = 0.0
        for j in range(N):
            acc += Grad_Out[i * N + j] * B[k * N + j]
        Grad_A[i * K + k] += acc
    for k, j in ti.ndrange(K, N):
        acc = 0.0
        for i in range(M):
            acc += A[i * K + k] * Grad_Out[i * N + j]
        Grad_B[k * N + j] += acc

@ti.kernel
def k_relu_backward(X: ti.types.ndarray(), Grad_Out: ti.types.ndarray(), Grad_X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        if X[i] > 0:
            Grad_X[i] += Grad_Out[i]

@ti.kernel
def k_matmul(A: ti.types.ndarray(), B: ti.types.ndarray(), C: ti.types.ndarray(), M: int, N: int, K: int):
    for i, j in ti.ndrange(M, N):
        acc = 0.0
        for k in range(K):
            acc += A[i * K + k] * B[k * N + j]
        C[i * N + j] = acc

@ti.kernel
def k_add_bias(X: ti.types.ndarray(), bias: ti.types.ndarray(), M: int, N: int):
    for i, j in ti.ndrange(M, N):
        X[i * N + j] += bias[j]

@ti.kernel
def k_relu_1d(X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        if X[i] < 0: X[i] = 0.0

@ti.kernel
def k_concat_dim(A: ti.types.ndarray(), B: ti.types.ndarray(), Out: ti.types.ndarray(), 
                 pre_dim: int, dim_a: int, dim_b: int, post_dim: int):
    dim_out = dim_a + dim_b
    for i, j, k in ti.ndrange(pre_dim, dim_out, post_dim):
        out_idx = (i * dim_out + j) * post_dim + k
        if j < dim_a:
            in_idx = (i * dim_a + j) * post_dim + k
            Out[out_idx] = A[in_idx]
        else:
            in_idx = (i * dim_b + (j - dim_a)) * post_dim + k
            Out[out_idx] = B[in_idx]

@ti.kernel
def k_silu_backward(X: ti.types.ndarray(), Grad_Out: ti.types.ndarray(), Grad_X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        x = X[i]
        sig = 1.0 / (1.0 + ti.exp(-x))
        Grad_X[i] += Grad_Out[i] * (sig * (1.0 + x * (1.0 - sig)))

@ti.kernel
def k_silu_1d(X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        x = X[i]
        X[i] = x / (1.0 + ti.exp(-x))

@ti.kernel
def k_gelu_tanh(X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        x = X[i]
        X[i] = 0.5 * x * (1.0 + ti.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

@ti.kernel
def k_gelu_tanh_backward(X: ti.types.ndarray(), Grad_Out: ti.types.ndarray(), Grad_X: ti.types.ndarray(), Total: int):
    for i in range(Total):
        # Precise GELU derivative
        x = X[i]
        x3 = x * x * x
        alpha = 0.7978845608
        beta = 0.044715
        z = alpha * (x + beta * x3)
        tanh_z = ti.tanh(z)
        sech2_z = 1.0 - tanh_z * tanh_z
        phi = 0.5 * (1.0 + tanh_z)
        term2 = 0.5 * x * sech2_z * alpha * (1.0 + 3.0 * beta * x * x)
        Grad_X[i] += Grad_Out[i] * (phi + term2)

@ti.kernel
def k_softmax_1d(X: ti.types.ndarray(), Out: ti.types.ndarray(), M: int, N: int):
    for i in range(M):
        max_val = -1e10
        for j in range(N):
            if X[i * N + j] > max_val: max_val = X[i * N + j]
        sum_exp = 0.0
        for j in range(N):
            val = ti.exp(X[i * N + j] - max_val)
            Out[i * N + j] = val
            sum_exp += val
        for j in range(N):
            Out[i * N + j] /= sum_exp

@ti.kernel
def k_softmax_backward(Out: ti.types.ndarray(), Grad_Out: ti.types.ndarray(), Grad_X: ti.types.ndarray(), M: int, N: int):
    for i in range(M):
        sum_grad_y = 0.0
        for j in range(N):
            sum_grad_y += Grad_Out[i * N + j] * Out[i * N + j]
        for j in range(N):
            y = Out[i * N + j]
            # dy_j / dx_k = y_j * (kron_jk - y_k)
            # sum_k (dL/dy_k * dy_k/dx_j)
            # = sum_k (dL/dy_k * y_k * (kron_kj - y_j))
            # = (dL/dy_j * y_j) - y_j * sum_k (dL/dy_k * y_k)
            Grad_X[i * N + j] += y * (Grad_Out[i * N + j] - sum_grad_y)

@ti.kernel
def k_rmsnorm_1d(X: ti.types.ndarray(), W: ti.types.ndarray(), Out: ti.types.ndarray(), M: int, N: int, eps: float, add_unit_offset: int):
    for i in range(M):
        rms_sq = 0.0
        for j in range(N):
            val = X[i * N + j]
            rms_sq += val * val
        rms = ti.sqrt(rms_sq / N + eps)
        for j in range(N):
            w = W[j]
            if add_unit_offset == 1: w += 1.0
            Out[i * N + j] = (X[i * N + j] / rms) * w

@ti.kernel
def k_rmsnorm_backward(X: ti.types.ndarray(), W: ti.types.ndarray(), Grad_Out: ti.types.ndarray(), 
                       Grad_X: ti.types.ndarray(), Grad_W: ti.types.ndarray(), 
                       M: int, N: int, eps: float, add_unit_offset: int):
    for i in range(M):
        rms_sq = 0.0
        for j in range(N):
            val = X[i * N + j]
            rms_sq += val * val
        rms = ti.sqrt(rms_sq / N + eps)
        sum_grad_w_x = 0.0
        for j in range(N):
            w = W[j]
            if add_unit_offset == 1: w += 1.0
            sum_grad_w_x += Grad_Out[i * N + j] * w * X[i * N + j]
        for j in range(N):
            w = W[j]
            if add_unit_offset == 1: w += 1.0
            grad_w_val = Grad_Out[i * N + j] * (X[i * N + j] / rms)
            Grad_W[j] += grad_w_val
            term1 = Grad_Out[i * N + j] * w / rms
            term2 = X[i * N + j] * sum_grad_w_x / (N * rms * rms * rms)
            Grad_X[i * N + j] += term1 - term2

@ti.kernel
def k_embedding_1d(Idx: ti.types.ndarray(), W: ti.types.ndarray(), Out: ti.types.ndarray(), B: int, L: int, D: int):
    for i, j, k in ti.ndrange(B, L, D):
        idx = Idx[i * L + j]
        Out[(i * L + j) * D + k] = W[idx * D + k]

@ti.kernel
def k_matmul_tiled(A: ti.types.ndarray(), B_tile: ti.types.ndarray(), C: ti.types.ndarray(), 
                   M: int, N_total: int, K_A: int, K_active: int, n_offset: int, n_tile: int, tile_size: int):
    for i, j_local in ti.ndrange(M, n_tile):
        acc = 0.0
        for k in range(K_active):
            acc += A[i * K_A + k] * B_tile[k * tile_size + j_local]
        C[i * N_total + (n_offset + j_local)] += acc

@ti.kernel
def k_matmul_tiled_grad_x(Grad_Out_Tile: ti.types.ndarray(), W_tile: ti.types.ndarray(), Grad_X: ti.types.ndarray(), M: int, K: int, n_tile: int, tile_size: int):
    for m, k in ti.ndrange(M, K):
        sum_val = 0.0
        for n in range(n_tile):
            sum_val += Grad_Out_Tile[m * n_tile + n] * W_tile[k * tile_size + n]
        Grad_X[m * K + k] += sum_val

@ti.kernel
def k_matmul_tiled_grad_w(X: ti.types.ndarray(), Grad_Out_Tile: ti.types.ndarray(), Grad_W_Tile: ti.types.ndarray(), M: int, K: int, n_tile: int, tile_size: int):
    for k, n in ti.ndrange(K, n_tile):
        sum_val = 0.0
        for m in range(M):
            sum_val += X[m * K + k] * Grad_Out_Tile[m * n_tile + n]
        Grad_W_Tile[k * tile_size + n] += sum_val

@ti.kernel
def k_transpose_2d(In: ti.types.ndarray(), Out: ti.types.ndarray(), H: int, W: int):
    for i, j in ti.ndrange(H, W):
        Out[j * H + i] = In[i * W + j]

@ti.kernel
def k_conv2d_1d(X: ti.types.ndarray(), W: ti.types.ndarray(), B: ti.types.ndarray(), Out: ti.types.ndarray(), 
                b: int, ic: int, oc: int, h: int, w: int, kh: int, kw: int):
    oh, ow = h - kh + 1, w - kw + 1
    for ib, ioc, ioh, iow in ti.ndrange(b, oc, oh, ow):
        acc = B[ioc]
        for iic, ikh, ikw in ti.ndrange(ic, kh, kw):
            acc += X[((ib * ic + iic) * h + (ioh + ikh)) * w + (iow + ikw)] * W[((ioc * ic + iic) * kh + ikh) * kw + ikw]
        Out[((ib * oc + ioc) * oh + ioh) * ow + iow] = acc

@ti.kernel
def k_mean_last_dim(X: ti.types.ndarray(), Out: ti.types.ndarray(), M: int, N: int):
    for i in range(M):
        acc = 0.0
        for j in range(N): acc += X[i * N + j]
        Out[i] = acc / N

@ti.kernel
def k_pow(X: ti.types.ndarray(), P: float, Total: int):
    for i in range(Total): X[i] = ti.pow(X[i], P)

@ti.kernel
def k_sqrt(X: ti.types.ndarray(), Total: int):
    for i in range(Total): X[i] = ti.sqrt(X[i])

@ti.kernel
def k_tanh(X: ti.types.ndarray(), Total: int):
    for i in range(Total): X[i] = ti.tanh(X[i])

@ti.kernel
def k_copy_offset(Src: ti.types.ndarray(), src_offset: int, Dst: ti.types.ndarray(), dst_offset: int, Total: int):
    for i in range(Total):
        Dst[dst_offset + i] = Src[src_offset + i]

@ti.kernel
def k_magnitude_norm(X: ti.types.ndarray(), Out: ti.types.ndarray(), slice_idx: int, ref_idx: int, B: int, L: int, D: int, eps: float):
    for b, l in ti.ndrange(B, L):
        mag_ref = 0.0
        mag_cur = 0.0
        for d in range(D):
            v_ref = X[((ref_idx * B + b) * L + l) * D + d]
            v_cur = X[((slice_idx * B + b) * L + l) * D + d]
            mag_ref += v_ref * v_ref
            mag_cur += v_cur * v_cur
        scale = ti.sqrt((mag_ref + eps) / (mag_cur + eps))
        for d in range(D):
            Out[((slice_idx * B + b) * L + l) * D + d] *= scale

@ti.kernel
def k_leaky_relu_1d(X: ti.types.ndarray(), Total: int, alpha: float):
    for i in range(Total):
        if X[i] < 0: X[i] *= alpha

@ti.kernel
def k_leaky_relu_backward(X: ti.types.ndarray(), Grad_Out: ti.types.ndarray(), Grad_X: ti.types.ndarray(), Total: int, alpha: float):
    for i in range(Total):
        if X[i] > 0:
            Grad_X[i] += Grad_Out[i]
        else:
            Grad_X[i] += Grad_Out[i] * alpha

@ti.kernel
def k_pool2d_1d(X: ti.types.ndarray(), Out: ti.types.ndarray(), b: int, c: int, h: int, w: int, s: int):
    oh, ow = h // s, w // s
    for ib, ic, ioh, iow in ti.ndrange(b, c, oh, ow):
        max_val = -1e10
        for ikh, ikw in ti.ndrange(s, s):
            val = X[((ib * c + ic) * h + (ioh * s + ikh)) * w + (iow * s + ikw)]
            if val > max_val: max_val = val
        Out[((ib * c + ic) * oh + ioh) * ow + iow] = max_val

@ti.kernel
def k_upsample2d_1d(X: ti.types.ndarray(), Out: ti.types.ndarray(), b: int, c: int, h: int, w: int, s: int):
    oh, ow = h * s, w * s
    for ib, ic, ioh, iow in ti.ndrange(b, c, oh, ow):
        Out[((ib * c + ic) * oh + ioh) * ow + iow] = X[((ib * c + ic) * h + (ioh // s)) * w + (iow // s)]

@ti.kernel
def k_rope(X: ti.types.ndarray(), Cos: ti.types.ndarray(), Sin: ti.types.ndarray(), B: int, L: int, H: int, D: int, pos_offset: int):
    for b, l, h, d_half in ti.ndrange(B, L, H, D // 2):
        pos = l + pos_offset
        idx1 = ((b * L + l) * H + h) * D + d_half * 2
        idx2 = idx1 + 1
        x1 = X[idx1]
        x2 = X[idx2]
        c = Cos[pos * D + d_half * 2]
        s = Sin[pos * D + d_half * 2]
        X[idx1] = x1 * c - x2 * s
        X[idx2] = x1 * s + x2 * c

@ti.kernel
def k_extract_slice(In: ti.types.ndarray(), Out: ti.types.ndarray(), slice_idx: int, B: int, L: int, D: int):
    for i in range(B * L * D):
        Out[i] = In[slice_idx * B * L * D + i]

@ti.kernel
def k_gaussian_topk(X: ti.types.ndarray(), Total: int, N: int, threshold: float):
    # Simplified placeholder for gaussian top-k
    for i in range(Total):
        if X[i] < threshold: X[i] = 0.0

@ti.kernel
def k_altup_predict(In: ti.types.ndarray(), Coefs: ti.types.ndarray(), Out: ti.types.ndarray(), S: int, B: int, L: int, D: int):
    for b, l, d in ti.ndrange(B, L, D):
        for out_s in range(S):
            acc = 0.0
            for in_s in range(S):
                c = Coefs[((b * L + l) * S + out_s) * S + in_s]
                acc += c * In[((in_s * B + b) * L + l) * D + d]
            Out[((out_s * B + b) * L + l) * D + d] = acc

@ti.kernel
def k_altup_correct(Predicts: ti.types.ndarray(), Act: ti.types.ndarray(), Coefs: ti.types.ndarray(), S: int, B: int, L: int, D: int, active_idx: int):
    for b, l, d in ti.ndrange(B, L, D):
        act_val = Act[(b * L + l) * D + d]
        for s in range(S):
            c = Coefs[(b * L + l) * S + s]
            Predicts[((s * B + b) * L + l) * D + d] += c * act_val

@ti.kernel
def k_altup_scale_correction(Corrected: ti.types.ndarray(), Scales: ti.types.ndarray(), S: int, B: int, L: int, D: int):
    for s, b, l, d in ti.ndrange(range(S), B, L, D):
        Corrected[((s * B + b) * L + l) * D + d] *= Scales[d]

@ti.kernel
def k_add_to_slices(Src: ti.types.ndarray(), Dst: ti.types.ndarray(), start_s: int, S: int, B: int, L: int, D: int):
    for s, b, l, d in ti.ndrange(range(start_s, S), B, L, D):
        Dst[((s * B + b) * L + l) * D + d] += Src[(b * L + l) * D + d]

@ti.kernel
def k_kv_cache_update(K_cache: ti.types.ndarray(), V_cache: ti.types.ndarray(), K_new: ti.types.ndarray(), V_new: ti.types.ndarray(), B: int, L: int, H: int, D: int, offset: int, max_len: int):
    for b, l, h, d in ti.ndrange(B, L, H, D):
        if offset + l < max_len:
            K_cache[((b * max_len + offset + l) * H + h) * D + d] = K_new[((b * L + l) * H + h) * D + d]
            V_cache[((b * max_len + offset + l) * H + h) * D + d] = V_new[((b * L + l) * H + h) * D + d]

@ti.kernel
def k_attention(Q: ti.types.ndarray(), K: ti.types.ndarray(), V: ti.types.ndarray(), Out: ti.types.ndarray(), B: int, L_new: int, L_total: int, H: int, D: int, scale: float, offset: int, window: int, softcap: float):
    for b, h, i in ti.ndrange(B, H, L_new):
        pos_i = i + offset
        for j in range(L_total):
            # Causal mask + window
            if j <= pos_i and (window == 0 or pos_i - j < window):
                attn_score = 0.0
                for d in range(D):
                    attn_score += Q[((b * L_new + i) * H + h) * D + d] * K[((b * L_total + j) * H + h) * D + d]
                attn_score *= scale
                # Apply softcap
                if softcap > 0:
                    attn_score = softcap * ti.tanh(attn_score / softcap)
                # This is a VERY simplified attention (missing softmax locally)
                # but matching the previous core.py structure for now.
                # In a real model, we'd do a proper softmax across j.
                pass

def warmup():
    """Compiles core kernels by running them with dummy data."""
    print("Pre-compiling Vulkan kernels...")
    dummy = ti.ndarray(ti.f32, shape=(1,))
    k_zero(dummy, 1)
    k_add(dummy, dummy, 1, 1)
    k_adam(dummy, dummy, dummy, dummy, 1.0, 0.9, 0.999, 1e-8, 1, 1)
    ti.sync()
    print("Kernels warmed up.")
