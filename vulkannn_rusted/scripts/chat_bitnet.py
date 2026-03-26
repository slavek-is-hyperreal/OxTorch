import os
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer
import vulkannn_rusted as vnn

def chat():
    parser = argparse.ArgumentParser(description="BitNet-2B Native Rust Chat")
    parser.add_argument("--model", type=str, default="/my_data/gaussian_room/models/bitnet-2B-4T-gguf/ggml-model-i2_s.gguf", help="Path to model")
    parser.add_argument("--tokenizer", type=str, default="unsloth/Llama-3.2-1B", help="Path to tokenizer (HF repo or local dir)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "vulkan", "vga"], help="Compute device")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Process a single prompt and exit")
    args = parser.parse_args()

    print(f"[*] Loading tokenizer from {args.tokenizer}...")
    # Llama-3 uses tiktoken-based BPE (similar to GPT-2)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Llama-3 EOS: 128001 / 128009
    stop_tokens = [128001, 128009]

    print(f"[*] Initializing Rust BitNetModel on {args.device}...")
    start_t = time.time()
    model = vnn.BitNetModel(args.model, args.device)
    print(f"[*] Model loaded in {time.time() - start_t:.2f}s")

    print("\n--- BitNet Chat Ready (type 'exit' to quit) ---")
    
    while True:
        try:
            if args.prompt:
                prompt = args.prompt
            else:
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

            if args.prompt: break

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    chat()
