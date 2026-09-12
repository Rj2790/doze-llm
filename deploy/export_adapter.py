"""Convert a harness resume-state file (backend.pt) into a standard PEFT adapter
folder (adapter_model.safetensors + adapter_config.json) that loads with
`PeftModel.from_pretrained(base_model, folder)`.

The resume state holds the trainable LoRA tensors under "trainable" with PEFT's
internal names (…lora_A.default.weight); PEFT's on-disk format drops the
adapter name. Optimizer and RNG state are not exported. Run with any Python
that has torch + safetensors + peft (the repo venv does not):

    python deploy/export_adapter.py <state_dir_or_backend.pt> <out_dir> [--base Qwen/Qwen3-4B]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends import training_utils as tu  # noqa: E402


def export(src: Path, out: Path, base: str) -> dict:
    import torch
    from safetensors.torch import save_file

    pt = src / "backend.pt" if src.is_dir() else src
    st = torch.load(pt, map_location="cpu", weights_only=False)
    trainable = st["trainable"]
    tensors = {k.replace(".default.", "."): v.detach().to(torch.float32).contiguous() for k, v in trainable.items()}
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base,
        "r": tu.LORA_RANK, "lora_alpha": tu.LORA_ALPHA, "lora_dropout": 0.0, "bias": "none",
        "target_modules": list(tu.LORA_TARGET_MODULES), "fan_in_fan_out": False, "inference_mode": True,
        "init_lora_weights": True, "use_rslora": False, "use_dora": False,
    }
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    info = {"source": str(pt), "n_tensors": len(tensors), "n_params": int(sum(v.numel() for v in tensors.values())),
            "seed": st.get("seed"), "optimizer_steps": None}
    try:
        opt = st.get("optimizer") or {}
        steps = [s.get("step") for s in opt.get("state", {}).values() if isinstance(s, dict) and "step" in s]
        if steps:
            info["optimizer_steps"] = int(max(float(x) if not hasattr(x, "item") else x.item() for x in steps))
    except Exception:
        pass
    (out / "export_info.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--base", default="Qwen/Qwen3-4B")
    a = ap.parse_args()
    info = export(Path(a.src), Path(a.out), a.base)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
