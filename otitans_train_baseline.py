import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Configuration & Path Sculpting
# We use mlabonne's abliterated Gemma 3 for the most sterile baseline possible.
MODEL_ID = "/home/paperscarecrow/Downloads/gemma-3-12b-it-abliterated"
# Open-Platypus is excellent for verifying STEM and logic retrieval.
DATASET_ID = "garage-bAInd/Open-Platypus"
OUTPUT_DIR = "./otitans_logic_baseline"

# Ensure the script finds the core orthogonal module
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
from otitans_surgery import inject_orthogonal_memory

class OTitansTrainer(Trainer):
    def __init__(self, ortho_lambda=0.15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_lambda = ortho_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        base_loss = outputs.loss

        ortho_penalty = 0.0
        for module in model.modules():
            if hasattr(module, 'get_orthogonal_penalty'):
                ortho_penalty += module.get_orthogonal_penalty()

        # We increase ortho_lambda to 0.15 to ensure the baseline is strictly isolated.
        total_loss = base_loss + (self.ortho_lambda * ortho_penalty)
        return (total_loss, outputs) if return_outputs else total_loss

def main():
    print(f"[*] Initializing Tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading Logic Dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")
    
    def format_and_tokenize(examples):
        # We format for standard Instruction/Input/Output logic
        texts = [f"Instruction: {i}\nInput: {inp}\nResponse: {o}" for i, inp, o in zip(examples['instruction'], examples['input'], examples['output'])]
        tokenized = tokenizer(texts, truncation=True, max_length=1024, padding="max_length")
        
        # Multimodal bypass: Explicitly tell Gemma 3 these are text-only tokens (0)
        tokenized["token_type_ids"] = [[0] * len(ids) for ids in tokenized["input_ids"]]
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_datasets = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)

    print("[*] Powering up the Kiln. Loading base weights...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Freeze the foundation
    for param in model.parameters():
        param.requires_grad = False

    print("[*] Commencing O-TITANS Sculpting on Query/Value vectors...")
    # Using Rank 16 for a more robust logic baseline.
    replaced_layers = inject_orthogonal_memory(model, target_modules=["q_proj", "v_proj"], rank=16, alpha=32.0)
    print(f"[*] Sculpting complete. {replaced_layers} O-TITANS modules active.")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8, # Increased for a smoother gradient on logic tasks
        learning_rate=2e-4,
        num_train_epochs=1, # One high-quality pass to avoid overfitting
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to="none"
    )

    trainer = OTitansTrainer(
        ortho_lambda=0.15,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    print("\n[*] Firing the Logic Baseline. Igniting the Forge...\n")
    trainer.train()

    # Save the Pure Adapter for GitHub/Hugging Face distribution
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    adapter_state_dict = {n: p.data for n, p in model.named_parameters() if p.requires_grad}
    torch.save(adapter_state_dict, os.path.join(OUTPUT_DIR, "otitans_logic_core_v1.pt"))
    print(f"[*] Sequence Terminated. Logic Core saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
