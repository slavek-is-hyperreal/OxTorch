import torch
import sys
import os
import psutil
import json
import time

# Add project root and Gemma directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Gemma')))

from gemma import config as gemma_config
from gemma import gemma3_model
from scripts import vnn_adapter

def monitor_ram():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    print(f"[RAM Monitor] Current Usage: {mem:.2f} MB")

def load_custom_config(config_path):
    with open(config_path, "r") as f:
        hf_config = json.load(f)
    
    text_config = hf_config["text_config"]
    
    # Map HF layer types to Gemma PyTorch enum
    attn_map = {
        "sliding_attention": gemma_config.AttentionType.LOCAL_SLIDING,
        "full_attention": gemma_config.AttentionType.GLOBAL
    }
    attn_types = [attn_map[t] for t in text_config["layer_types"]]
    
    from gemma.siglip_vision import config as siglip_vision_config
    
    # Fill vision config (even if it's MobileNet vs SigLip mismatch, we need it to pass init)
    v_hf = hf_config.get("vision_config", {})
    vision_config = siglip_vision_config.SiglipVisionModelConfig(
        num_hidden_layers=v_hf.get("num_hidden_layers", 27),
        embedding_dim=v_hf.get("hidden_size", 1152),
        num_attention_heads=v_hf.get("num_attention_heads", 16),
        intermediate_size=v_hf.get("intermediate_size", 4304),
    )

    # Create GemmaConfig
    model_config = gemma_config.GemmaConfig(
        architecture=gemma_config.Architecture.GEMMA_3,
        num_hidden_layers=text_config["num_hidden_layers"],
        hidden_size=text_config["hidden_size"],
        intermediate_size=text_config["intermediate_size"][0],
        num_attention_heads=text_config["num_attention_heads"],
        num_key_value_heads=text_config["num_key_value_heads"],
        head_dim=text_config["head_dim"],
        vocab_size=text_config["vocab_size"],
        rms_norm_eps=text_config["rms_norm_eps"],
        final_logit_softcapping=text_config.get("final_logit_softcapping"),
        attn_logit_softcapping=text_config.get("attn_logit_softcapping"),
        attn_types=attn_types,
        sliding_window_size=text_config.get("sliding_window", 1024),
        use_qk_norm=True,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        tokenizer="gemma-weights/tokenizer.model",
        vision_config=vision_config,
        rope_wave_length={
            gemma_config.AttentionType.LOCAL_SLIDING: text_config.get("rope_local_base_freq", 10000.0),
            gemma_config.AttentionType.GLOBAL: text_config.get("rope_theta", 1000000.0)
        },
        rope_scaling_factor=text_config.get("rope_scaling", {}).get("factor", 1.0) if text_config.get("rope_scaling") else 8, # Gemma 3n usually uses 8
        dtype='float32'
    )

    return model_config

def main():
    print("--- Gemma 3n OOM-Safe Chat Demo (VNN Rusted) ---")
    monitor_ram()

    # 1. Load weights map and custom config
    weights_map_path = "/vectorlegis_ssd_pool/vnn_cache/gemma-3n-E4B-it-f32/vnn_weights_map.json"
    hf_config_path = "gemma-weights/config.json"
    
    if not os.path.exists(hf_config_path):
        print(f"Error: {hf_config_path} not found. Ensure weights are downloaded.")
        return

    model_config = load_custom_config(hf_config_path)
    print(f"Loaded config: hidden_size={model_config.hidden_size}, layers={model_config.num_hidden_layers}")

    # 2. Initialize Model (Empty RAM)
    print("\n[1/3] Initializing Gemma 3n structure (Empty RAM)...")
    device = torch.device('cpu')
    with torch.device('meta'):
        model = gemma3_model.Gemma3ForMultimodalLM(model_config)
    
    # Move to CPU but keep as 'empty' (meta-device doesn't have storage)
    model = model.to_empty(device=device)
    monitor_ram()

    # 3. Apply VNN Bridge (SSD Mapping)
    print("\n[2/3] Applying VNN Rusted Bridge (SSD Mapping & PLE)...")
    vnn_adapter.bridge_to_vnn(model, weights_map_path)
    monitor_ram()

    # 4. Chat Loop
    print("\n[3/3] Starting Inference Loop (OOM-Safe)...")
    
    prompt = "Explain the concept of 'Infinite Memory' in VNN Rusted engine."
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    print(f"\nUser: {prompt}")
    
    start_time = time.time()
    try:
        # Note: generate() internal calls model.forward which we've patched to propagate input_ids
        result = model.generate(
            [[formatted_prompt]],
            device,
            output_len=100,
            temperature=None # Greeddy for testing
        )
        end_time = time.time()
        
        print(f"\nGemma: {result}")
        print(f"\nInference time: {end_time - start_time:.2f}s")
        
    except Exception as e:
        print(f"\nInference failed: {e}")
        import traceback
        traceback.print_exc()

    monitor_ram()
    print("\nDemo complete. Weights remained on SSD throughout.")

if __name__ == "__main__":
    main()
