from pathlib import Path
from types import SimpleNamespace
from scripts.run_qwen_image_2_1_lora import build_config, preflight

def args(tmp_path: Path):
    ds=tmp_path/"data"; ds.mkdir()
    for i in range(15):
        (ds/f"{i}.png").write_bytes(b"x")
        (ds/f"{i}.txt").write_text("person portrait",encoding="utf-8")
    tk=tmp_path/"ai-toolkit"; tk.mkdir(); (tk/"run.py").write_text("",encoding="utf-8")
    return SimpleNamespace(dataset=ds,output_dir=tmp_path/"out",model="Qwen/Qwen-Image-2.1",
      ai_toolkit=str(tk),name="test",trigger_word="TOK",resolution=1024,rank=16,alpha=16,
      steps=50,grad_accum=1,lr=1e-4,save_every=25)

def test_preflight_accepts_captioned_character_dataset(tmp_path):
    a=args(tmp_path); report=preflight(a)
    assert report["ok"] is True and report["images"]==15 and report["captions"]==15

def test_config_uses_qwen_image_2_and_freezes_text_encoder(tmp_path):
    cfg=build_config(args(tmp_path)); proc=cfg["config"]["process"][0]
    assert proc["model"]["arch"]=="qwen_image_2"
    assert proc["train"]["train_text_encoder"] is False
    assert proc["train"]["gradient_checkpointing"] is True
    assert proc["datasets"][0]["cache_latents_to_disk"] is True
