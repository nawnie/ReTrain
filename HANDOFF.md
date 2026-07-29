# ReTrain Handoff

## Current product boundary

ReTrain is a standalone local training workbench. Its FastAPI backend plans and
tracks runs; the React frontend presents readiness gates, local-model and
dataset discovery, VRAM estimates, receipts, logs, scalar summaries, and an
embedded TensorBoard view. It may discover an explicitly configured external
MoK dataset, but it does not become part of MoK or AIWF Studio.

## Execution truth

- `scripts/run_posttrain_bakeoff.py` executes full SFT, LoRA, and QLoRA for
  causal/prompt-helper decoder LMs.
- `scripts/run_text_target_training.py` executes full Seq2Seq/T5 and masked-LM
  text-encoder fine-tuning.
- CLIP, BLIP, component encoders, DoRA, adapter stacking, expert routing, and
  dense growth are not executable paths in this checkout.
- `dryRun: true` validates selected paths and writes a receipt without loading
  model weights. Real training additionally requires confirmation.

## Local configuration

The public repository intentionally contains no machine-specific model,
wheelhouse, or external-project locations. Configure optional local sources
with the environment variables documented in `README.md`; `models/` is the
portable default location for already-downloaded base models.

## Historical verification boundary

Earlier local checks recorded successful static compilation, dataset
validation, frontend build, loopback API health, and a dry-run receipt on a
then-current workstation. Those receipts establish prior development evidence;
they are not a claim that a fresh checkout has been installed or trained on a
new machine. Re-run the checks in `README.md` before making an installation,
CUDA, training, or performance claim.

## Next implementation work

1. Add an explicit adapter-package format and loader contract.
2. Add a DoRA path only with matching configuration, tests, and a measured
   smoke result.
3. Add paired-data runners before enabling CLIP or BLIP targets.
4. Run a confirmed, bounded local training and evaluation pass before claiming
   model quality or adapter value.
