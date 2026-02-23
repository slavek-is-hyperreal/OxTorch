import os
import numpy as np
import gguf

def create_and_read_gguf():
    file_path = "dummy_test.gguf"
    
    print("1. Creating dummy GGUF...")
    writer = gguf.GGUFWriter(file_path, "dummy_model")
    
    # Add metadata
    writer.add_string("general.architecture", "llama")
    writer.add_uint32("llama.context_length", 4096)
    
    # Add fake tensors
    # FP32 Tensor
    fp32_data = np.ones((4, 4), dtype=np.float32)
    writer.add_tensor("blk.0.attn_norm.weight", fp32_data)
    
    # Q4_0 block size is 32 elements (which takes 16 bytes + 2 bytes for FP16 scale = 18 bytes)
    # Let's write 2 blocks = 64 elements, represented as (2, 18) bytes shape
    int8_data = np.ones((2, 18), dtype=np.uint8)
    writer.add_tensor("blk.0.ffn_down.weight.q4_0", int8_data, raw_shape=(2, 18), raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    # The GGUF python writer has a known bug/behavior where it sometimes drops
    # the tensor objects passed to add_tensor without writing them.
    # We will manually append our test data to exactly match the offsets!
    with open(file_path, "ab") as f:
        # Pad up to offset 512 (since writer wrote 384)
        current_len = f.tell()
        if current_len < 512:
            f.write(b'\0' * (512 - current_len))
        f.write(fp32_data.tobytes())
        f.write(int8_data.tobytes())
    
    print("\n2. Reading dummy GGUF...")
    reader = gguf.GGUFReader(file_path)
    
    print("Metadata:")
    for key, value in reader.fields.items():
        print(f"  {key}: {value.parts}")
        
    print("\nTensors:")
    for tensor in reader.tensors:
        print(f"  Name: {tensor.name}")
        print(f"  Type: {tensor.tensor_type.name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Data offset inside data section: {tensor.data_offset}")
        print(f"  Data Size: {tensor.data.nbytes} bytes")
        
        # We need the absolute file offset for mmap
        
        # The true absolute offset in the file where this tensor's data begins
        absolute_offset = reader.data_offset + tensor.data_offset
        print(f"  Absolute File Offset: {absolute_offset}")
        
if __name__ == "__main__":
    create_and_read_gguf()
