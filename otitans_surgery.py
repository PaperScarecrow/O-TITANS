import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Map our absolute paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "mlabonne/gemma-3-12b-it-abliterated"

# Ensure the script can find our custom core module
sys.path.append(PROJECT_DIR)
from otitans_core import OLoRALinear

def inject_orthogonal_memory(model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0):
    """
    Phase 4: The Surgeon's Scalpel.
    Recursively hunts for target linear layers and replaces them with the OTITANS shield.
    """
    injected_count = 0
    
    # We convert named_modules() to a list so we can modify the model dictionary while iterating
    for name, module in list(model.named_modules()):
        
        # Check if the current module ends with any of our targets (e.g., 'layers.0.self_attn.q_proj')
        if any(name.endswith(target) for target in target_modules):
            
            # We only perform surgery on standard Linear layers to prevent breaking Layernorms
            if isinstance(module, nn.Linear):
                
                # Instantiate our custom Orthogonal wrapper around the existing frozen layer
                wrapped_layer = OLoRALinear(base_layer=module, rank=rank, alpha=alpha)
                
                # PyTorch Surgery: Find the parent module and physically overwrite the child attribute
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[-1]
                
                # get_submodule is a clean PyTorch 1.9+ way to fetch nested parents
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, wrapped_layer)
                
                injected_count += 1
                
    return injected_count

def main():
    print(f"[*] Opening the Forge. Loading base weights from:\n    {MODEL_PATH}")
    
    # Load the base model strictly in bfloat16 to preserve our 50GB VRAM headroom
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    
    print("[*] Base architecture loaded. Freezing all 12 Billion primary parameters...")
    # This is slightly redundant since OLoRALinear does this, but it is excellent practice 
    # to enforce a total freeze before making any structural changes.
    for param in model.parameters():
        param.requires_grad = False
        
    print("[*] Commencing OTITANS surgery. Injecting Orthogonal LoRA into Attention vectors...")
    
    # We target Query and Value projections. 
    # In recurrent memory theory, Queries ask for memory, Values store the memory.
    targets = ["q_proj", "v_proj"]
    replaced_layers = inject_orthogonal_memory(model, target_modules=targets, rank=8, alpha=16.0)
    
    print(f"[*] Surgery complete. Successfully grafted {replaced_layers} OLoRALinear modules.")
    
    # Verify exactly how much of the network is actually trainable now
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("-" * 50)
    print(f"[*] Total Parameters:     {total_params:,}")
    print(f"[*] Trainable OTITANS:    {trainable_params:,}")
    print(f"[*] Optimization Footprint: {(trainable_params / total_params) * 100:.4f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()
