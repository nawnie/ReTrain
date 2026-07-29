# Changelog

## Unreleased

- Reframed ReTrain as a local training workbench with planning, dry-run,
  receipt, and TensorBoard workflows rather than a generic dashboard.
- Documented the current engines accurately: causal-LM full SFT/LoRA/QLoRA,
  Seq2Seq/T5 full fine-tuning, and masked-LM text-encoder full fine-tuning.
- Marked DoRA, adapter stacking/routing, expert packs, and dense growth as
  planned work rather than current execution modes.
- Removed published workstation paths from model discovery, runner presets,
  installer wheelhouse handling, historical handoff notes, and setup docs.
- Added public-release privacy and repository-boundary documentation.
- Updated PostCSS from `^8.5.16` to `^8.5.18`; a production dependency audit
  now reports zero vulnerabilities.
