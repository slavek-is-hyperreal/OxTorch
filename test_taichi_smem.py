import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan)

BLOCK_SIZE = 16 

@ti.kernel
def matmul_shared(A: ti.types.ndarray(), B: ti.types.ndarray(), C: ti.types.ndarray(), M: int, N: int, K: int):
    # Padding the thread grid to multiples of BLOCK_SIZE
    # We use BLOCK_SIZE = 16 for better compatibility with 1GB VRAM (less shared memory usage per block)
    M_padded = (M + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    N_padded = (N + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    num_blocks_i = M_padded // BLOCK_SIZE
    num_blocks_j = N_padded // BLOCK_SIZE
    
    ti.loop_config(block_dim=BLOCK_SIZE * BLOCK_SIZE)
    for linear_idx in range(num_blocks_i * num_blocks_j * BLOCK_SIZE * BLOCK_SIZE):
        block_idx = linear_idx // (BLOCK_SIZE * BLOCK_SIZE)
        thread_idx = linear_idx % (BLOCK_SIZE * BLOCK_SIZE)
        
        block_i = block_idx // num_blocks_j
        block_j = block_idx % num_blocks_j
        
        i_local = thread_idx // BLOCK_SIZE
        j_local = thread_idx % BLOCK_SIZE
        
        i_global = block_i * BLOCK_SIZE + i_local
        j_global = block_j * BLOCK_SIZE + j_local
        
        # Adding + 1 to avoid bank conflicts!
        sA = ti.simt.block.SharedArray((BLOCK_SIZE, BLOCK_SIZE + 1), ti.f32)
        sB = ti.simt.block.SharedArray((BLOCK_SIZE, BLOCK_SIZE + 1), ti.f32)
        
        acc = 0.0
        # Iterate over tiles in the K dimension
        for k_block in range((K + BLOCK_SIZE - 1) // BLOCK_SIZE):
            
            # Collaborative Loading: Each thread loads 1 element of A and 1 of B
            k_global_a = k_block * BLOCK_SIZE + j_local
            if i_global < M and k_global_a < K:
                sA[i_local, j_local] = A[i_global * K + k_global_a]
            else:
                sA[i_local, j_local] = 0.0
                
            k_global_b = k_block * BLOCK_SIZE + i_local
            if k_global_b < K and j_global < N:
                sB[i_local, j_local] = B[k_global_b * N + j_global]
            else:
                sB[i_local, j_local] = 0.0
                
            ti.simt.block.sync() # Wait for all threads to finish loading
            
            # Compute partial dot product from shared memory
            for k in range(BLOCK_SIZE):
                acc += sA[i_local, k] * sB[k, j_local]
                
            ti.simt.block.sync() # Wait before overwriting shared memory in next iteration
            
        # Only write back via threads that are inside the actual matrix boundary
        if i_global < M and j_global < N:
            C[i_global * N + j_global] = acc

def verify():
    M, N, K = 30, 40, 50
    A = ti.ndarray(ti.f32, shape=M*K)
    B = ti.ndarray(ti.f32, shape=K*N)
    C = ti.ndarray(ti.f32, shape=M*N)

    A.from_numpy(np.ones(M*K, dtype=np.float32) * 2.0)
    B.from_numpy(np.ones(K*N, dtype=np.float32) * 3.0)

    matmul_shared(A, B, C, M, N, K)
    
    np_res = C.to_numpy()
    print(f"Sample corner: {np_res[:5]}")
    
    # 2.0 * 3.0 * 50 = 300.0
    assert np.all(np_res == 300.0)
    print("Correctness verified for arbitrary dimensions!")

if __name__ == "__main__":
    verify()
