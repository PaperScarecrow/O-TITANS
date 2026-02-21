import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 1. Map Universal Paths
# Using relative paths ensures the repository is portable across hardware environments.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Users should point this to their local mlabonne/gemma-3-12b-it-abliterated directory
MODEL_PATH = "mlabonne/gemma-3-12b-it-abliterated" 
ADAPTER_PATH = os.path.join(PROJECT_DIR, "otitans_adapter.pt")

# Bring in the O-TITANS core sculpting logic
sys.path.append(PROJECT_DIR)
from otitans_surgery import inject_orthogonal_memory

def main():
    print(f"[*] Initializing O-TITANS Inference. Loading foundation: {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("[*] Reconstructing Orthogonal Architecture...")
    # Replicating the sculpting parameters: q_proj and v_proj isolation.
    targets = ["q_proj", "v_proj"]
    replaced_layers = inject_orthogonal_memory(model, target_modules=targets, rank=8, alpha=16.0)
    
    # Ensure all grafted layers are aligned to the GPU bus.
    model.to(torch.bfloat16)
    
    print(f"[*] Loading O-TITANS memory states from {ADAPTER_PATH}...")
    # Weights_only=True is a critical security protocol for public releases.
    adapter_state_dict = torch.load(ADAPTER_PATH, map_location=model.device, weights_only=True)
    
    # Strict=False allows us to graft the LoRA weights without touching the 12B base parameters.
    model.load_state_dict(adapter_state_dict, strict=False)
    model.eval()

    print("[*] Neural Graft Complete. Terminal Online.")
    print("-" * 50)
    
    # 2. System Prompt Definition
    # This allows the user to define the model's baseline behavior.
    system_prompt = "You are a helpful, objective assistant. Your logic is augmented by O-TITANS memory retrieval."
    chat_history = [{"role": "system", "content": system_prompt}]
    
    while True:
        try:
            user_input = input("\nUser: ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            chat_history.append({"role": "user", "content": user_input})
            
            prompt = tokenizer.apply_chat_template(
                chat_history, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            print(f"\nAssistant: ", end="", flush=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True, # Speed fix: O-LoRAs support native caching.
                    streamer=streamer,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            input_length = inputs.input_ids.shape[1]
            response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            chat_history.append({"role": "assistant", "content": response_text})
            print("\n") 

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
