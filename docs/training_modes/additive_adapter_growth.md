# Additive Adapter Growth

**Status:** planned training mode  
**Alternate name:** Frozen-Core Additive Expert Training

Additive Adapter Growth keeps the base model frozen and trains additional adapter or expert weights beside it. The goal is to expand capability without overwriting the original model.

```text
1.5B frozen base model
+ trainable adapter/expert weights
= expanded 1.6B-2B total model package
```

This is not full dense model expansion. The base architecture remains stable while ReTrain produces modular add-ons that can be saved, swapped, stacked, routed, or merged later.

## Current implementation boundary

- The causal-LM runner implements full SFT, LoRA, and QLoRA.
- LoRA and QLoRA can produce separate adapter artifacts; full SFT writes a
  fine-tuned model output.
- The text-side runner currently performs full Seq2Seq or masked-LM tuning; it
  does not implement adapter tuning for those target types.
- DoRA, adapter stacking/routing, expert packs, and dense growth are planned,
  not executable modes in this checkout.

## Proposed additive path

- Load a 1.5B-class base model as a stable frozen core.
- Start with the existing LoRA or QLoRA path and save adapter artifacts
  separately from base weights.
- Add DoRA or another adapter family only after its configuration, save/load
  contract, tests, and bounded run evidence are present.
- Keep full dense architecture growth behind an experimental gate.

## Adapter Growth Stages

Use staged growth so training does not jump straight into oversized adapters.

```text
Stage 1: 16M adapter
Stage 2: 64M adapter
Stage 3: 128M adapter
Stage 4: 250M+ adapter/expert
Stage 5: experimental dense model growth
```

## Capability Packs

The additive model path should support specialized packs, including:

- Code adapter
- Tool-use adapter
- Routing adapter
- Atlas/card reasoning adapter
- Style/personality adapter
- Safety/alignment adapter

## Guardrails

- Label this clearly as additive adapter training, not full model pretraining.
- Do not imply that a 1.5B dense model has become a native 2B dense model.
- Require explicit confirmation before any real VRAM-heavy training run.
- Keep base weights, training outputs, and large adapter artifacts out of git.
- Treat true 1.5B-to-2B dense growth as research mode requiring architecture changes and continued pretraining.
