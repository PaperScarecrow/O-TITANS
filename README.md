# O-TITANS: Orthogonal Tensors for Independent Task Alignment

The standard approach to training and merging local LLM adapters usually involves a brute-force overwrite of the base model's logic. If you later try to merge multiple LoRAs (like a "Physics" expert and a "Poetry" expert) into a base model simultaneously, their weight matrices collide. The model attempts to average them out, resulting in destructive interference and a lobotomized output.

To solve this, we've developed a new approach to recurrent memory adapters. Building on Google's TITANS architecture and heavily inspired by **ffurfaro's** TPTT "titanesque" methods, we've created **O-TITANS** (Orthogonal Tensors for Independent Task Alignment). 

## The Core Concept: Mathematical Orthogonality

Standard orthogonal LoRAs often sweep the entire network blindly, which can degrade the foundational logic of the model. 

O-TITANS adapters are trained with a strict orthogonal penalty applied to very specific layers. Because of this penalty, their learned weights are forced into completely independent vector subspaces. This means that when you eventually merge or hot-swap these weights, they do not destructively interfere with the core model's latent space—or with other orthogonal adapters. You retain the foundation's logic while cleanly injecting the new skill.

## The O-TITANS Sculpting (Why it works)

Instead of targeting every linear layer, O-TITANS targets specific attention vectors based on recurrent memory theory: **Queries ask for memory, and Values store it.** By isolating the orthogonal penalty exclusively to the `q_proj` and `v_proj` linear layers during training, we create an independent memory retrieval system while explicitly protecting the core normalization pathways and baseline reasoning of the foundation model. 

*(Note: We've found the Gemma 3 architecture to be the most receptive blank slate for this methodology. Qwen3 and Llama 3.2 currently require heavy tokenizer/chat-template fighting to prevent semantic collapse during the surgical injection).*

## The "Frontal Lobe" Baseline

For the ultimate sterile baseline to train and test these adapters, we exclusively utilize **mlabonne's** BF16 Gemma 3 abliterated models. Because his `minosv1` process effectively zeroes out the alignment vectors, it provides a perfectly clean foundation. It takes the O-TITANS outputs and synthesizes them into a unified, coherent response without injecting corporate alignment noise or fighting the new orthogonal weights.

## Links & Assets
* **O-TITANS Gemma 3 Adapters (Proof of Concept):** [Insert Hugging Face Link Here]
* **Training Scripts & Surgery Methodology:** [Insert GitHub Repo Link Here]

## Credits & Resources

A massive credit to the foundational work that made this possible:
* **[ffurfaro](https://huggingface.co/ffurfaro)** for the TPTT "titanesque" methodologies that inspired the titanized-lora structural approach.
* **[mlabonne](https://huggingface.co/mlabonne)** for the BF16 Gemma-3-abliteration models. The zeroed vectors from his `minosv1` process are what make the underlying synthesis actually work without semantic contamination.

---
*We would love to hear how the community approaches stress-testing this, or if anyone has success porting the `q_proj`/`v_proj` orthogonal penalty to Qwen or Llama architectures without the tokenization breakdowns we experienced.*
