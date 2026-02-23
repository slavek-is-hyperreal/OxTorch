import numpy as np
from vulkan_nn_lib.tensor_store import GGUFTensorStore

def test_gguf_store():
    gguf_path = "dummy_test.gguf"
    
    # Needs dummy file from previous script to exist
    store = GGUFTensorStore(gguf_path)
    print("Found tensors:", list(store.tensors.keys()))
    
    # Read FP32 (4x4 = 16 floats = 64 bytes)
    fp32_raw = store.get_tensor("blk.0.attn_norm.weight")
    print(f"FP32 Raw Bytes Shape: {fp32_raw.shape}")
    
    # Cast raw bytes back to fp32 logically just to check
    fp32_view = fp32_raw.view(np.float32).reshape(4, 4)
    print(f"FP32 Extracted: {fp32_view[0]}")
    assert np.all(fp32_view == 1.0)
    
    # Read Q4_0 (2 blocks = 36 bytes)
    q40_raw = store.get_tensor("blk.0.ffn_down.weight.q4_0")
    print(f"Q4_0 Raw Bytes Shape: {q40_raw.shape}")
    assert q40_raw.shape == (36,)
    
    print("GGUFTensorStore works perfectly!")

if __name__ == "__main__":
    test_gguf_store()
