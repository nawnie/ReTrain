# Qwen Image 2.1 character LoRA

ReTrain supports a first executable Qwen Image 2.1 character-LoRA path through the upstream MIT-licensed ostris/ai-toolkit engine. ReTrain owns dataset validation, the character preset, generated training configuration, receipts/telemetry integration, and invocation; AI Toolkit supplies the architecture-specific diffusion implementation.

## Why this path
Qwen Image 2.1 is not a causal-LM PEFT job. Reusing ReTrain's existing LLM runner would be incorrect. The upstream engine has a dedicated `qwen_image_2` architecture implementation.

## Dataset
Use 15+ images for a smoke dataset and preferably 40-60 reviewed images for the first character run. Every image must have a same-stem `.txt` caption. Keep validation/holdout images outside the training folder.

## Windows example
```powershell
$env:RETRAIN_AI_TOOLKIT_ROOT="F:\\ai-toolkit"
python scripts/run_qwen_image_2_1_lora.py --dataset F:\\datasets\\character --output-dir F:\\training\\qwen21-character --dry-run
```
Remove `--dry-run` only after the generated config and preflight are reviewed.

Defaults: 1024px, batch 1, rank/alpha 16/16, BF16 compute, AdamW8bit, gradient checkpointing, latent/text-embedding caching, quantized base/text encoder, and low-VRAM mode. Text encoder is frozen.

This is an initial low-VRAM integration. It does not claim a completed training run or measured VRAM on a specific workstation until a local smoke test is recorded.
