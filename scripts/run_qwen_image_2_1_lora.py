from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLKIT = Path(os.environ.get("RETRAIN_AI_TOOLKIT_ROOT", ROOT / "external" / "ai-toolkit"))

def image_count(root: Path) -> int:
    exts={".png",".jpg",".jpeg",".webp",".bmp"}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)

def caption_coverage(root: Path) -> tuple[int,int]:
    exts={".png",".jpg",".jpeg",".webp",".bmp"}
    images=[p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return len(images), sum(1 for p in images if p.with_suffix(".txt").exists())

def build_config(a: argparse.Namespace) -> dict[str,Any]:
    return {
      "job":"extension","config":{"name":a.name,"process":[{
        "type":"sd_trainer","training_folder":str(a.output_dir),
        "device":"cuda:0","trigger_word":a.trigger_word,
        "network":{"type":"lora","linear":a.rank,"linear_alpha":a.alpha},
        "save":{"dtype":"bf16","save_every":a.save_every,"max_step_saves_to_keep":3},
        "datasets":[{"folder_path":str(a.dataset),"caption_ext":"txt",
          "caption_dropout_rate":0.05,"shuffle_tokens":False,
          "cache_latents_to_disk":True,"resolution":[a.resolution]}],
        "train":{"batch_size":1,"steps":a.steps,"gradient_accumulation_steps":a.grad_accum,
          "train_unet":True,"train_text_encoder":False,"gradient_checkpointing":True,
          "noise_scheduler":"flowmatch","optimizer":"adamw8bit","lr":a.lr,
          "dtype":"bf16","cache_text_embeddings":True},
        "model":{"name_or_path":a.model,"arch":"qwen_image_2",
          "quantize":True,"quantize_te":True,"low_vram":True}
      }]}
    }

def preflight(a: argparse.Namespace) -> dict[str,Any]:
    total,captions=caption_coverage(a.dataset)
    toolkit=Path(a.ai_toolkit)
    run_py=toolkit/"run.py"
    issues=[]
    if total < 15: issues.append("Character dataset has fewer than 15 images.")
    if captions != total: issues.append(f"Caption coverage is {captions}/{total}; each training image should have a matching .txt caption.")
    if not run_py.exists(): issues.append(f"AI Toolkit run.py not found at {run_py}. Set RETRAIN_AI_TOOLKIT_ROOT or --ai-toolkit.")
    if a.rank < 1: issues.append("LoRA rank must be positive.")
    return {"ok":not issues,"images":total,"captions":captions,"issues":issues,
      "toolkit":str(toolkit),"model":a.model,"arch":"qwen_image_2",
      "profile":"qwen_image_2_1_character","resolution":a.resolution,"steps":a.steps}

def main() -> int:
    p=argparse.ArgumentParser(description="ReTrain Qwen Image 2.1 character LoRA runner")
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen-Image-2.1")
    p.add_argument("--ai-toolkit",default=str(DEFAULT_TOOLKIT))
    p.add_argument("--name",default="qwen-image-2.1-character")
    p.add_argument("--trigger-word",default="")
    p.add_argument("--resolution",type=int,default=1024)
    p.add_argument("--rank",type=int,default=16)
    p.add_argument("--alpha",type=int,default=16)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--grad-accum",type=int,default=1)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--save-every",type=int,default=250)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--write-config",type=Path)
    a=p.parse_args()
    a.dataset=a.dataset.expanduser().resolve(); a.output_dir=a.output_dir.expanduser().resolve()
    report=preflight(a); print(json.dumps({"preflight":report},indent=2))
    if not report["ok"]: return 2
    cfg=build_config(a)
    config_path=a.write_config or (a.output_dir/"qwen_image_2_1_character.json")
    config_path.parent.mkdir(parents=True,exist_ok=True)
    config_path.write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
    if a.dry_run:
        print(json.dumps({"dryRun":True,"config":str(config_path)},indent=2)); return 0
    cmd=[sys.executable,str(Path(a.ai_toolkit)/"run.py"),str(config_path)]
    return subprocess.call(cmd,cwd=str(Path(a.ai_toolkit)))

if __name__=="__main__": raise SystemExit(main())
