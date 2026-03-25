import vulkannn_rusted as vnn
import torch
import os
from transformers import AutoTokenizer

def test_native_inference():
    model_path = "/my_data/gaussian_room/models/bitnet-2B-ternary"
    device = "cpu" # Test CPU first for easier debugging
    
    print(f"Loading native BitNetModel from {model_path} on {device}...")
    model = vnn.BitNetModel(model_path, device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = "What is 1.58-bit quantization?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids[0].tolist()
    
    print(f"Prompt: {prompt}")
    print(f"Generating tokens...")
    
    # Generate 20 tokens
    output_ids = model.generate(input_ids, 20)
    
    decoded = tokenizer.decode(output_ids)
    print(f"\nGenerated Output:\n{decoded}")

if __name__ == "__main__":
    test_native_inference()
