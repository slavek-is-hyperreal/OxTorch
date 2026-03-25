import os
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer
import vulkannn_rusted as vnn

def chat():
    parser = argparse.ArgumentParser(description="BitNet-2B Native Rust Chat")
    parser.add_argument("--model", type=str, default="/my_data/gaussian_room/models/bitnet-2B-ternary/", help="Path to model directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "vulkan", "vga"], help="Compute device")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"[*] Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False, fix_mistral_regex=True)
    
    # Llama-3 EOS: 128001 / 128009
    stop_tokens = [128001, 128009]

    print(f"[*] Initializing Rust BitNetModel on {args.device}...")
    start_t = time.time()
    model = vnn.BitNetModel(args.model, args.device)
    print(f"[*] Model loaded in {time.time() - start_t:.2f}s")

    print("\n--- BitNet Chat Ready (type 'exit' to quit) ---")
    
    while True:
        try:
            prompt = input("\nUser > ")
            if prompt.lower() in ["exit", "quit"]: break
            
            # Format prompt (standard Llama-3 chat template or simple text)
            # For now, keep it simple
            inputs = tokenizer.encode(prompt, add_special_tokens=True)
            
            print("AI > ", end="", flush=True)
            
            # Use the Rust generate loop
            start_gen = time.time()
            output_ids = model.generate(inputs, args.max_new_tokens)
            
            # Decode the NEW tokens only
            new_ids = output_ids[len(inputs):]
            response = tokenizer.decode(new_ids, skip_special_tokens=True)
            print(response)
            
            dt = time.time() - start_gen
            tps = len(new_ids) / dt if dt > 0 else 0
            print(f"\n[Stats: {len(new_ids)} tokens, {dt:.2f}s, {tps:.2f} tok/s]")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    chat()
