#!/usr/bin/env python3
import argparse
import glob
import shutil
import json
import os
import re
import sys
from typing import Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def read_yaml_keys(yaml_path: str) -> tuple[str, str]:
    """Read only the required keys from YAML. Falls back to a tiny line parser if PyYAML is missing."""
    if yaml is not None:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML at {yaml_path} does not contain a top-level mapping.")

        base_model = data.get("model_name_or_path")
        output_dir = data.get("output_dir")
    else:
        base_model = None
        output_dir = None
        with open(yaml_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # simple key: value extraction (no nested support needed for our keys)
                if line.startswith("model_name_or_path:") and base_model is None:
                    base_model = line.split(":", 1)[1].strip().strip("'\"")
                elif line.startswith("output_dir:") and output_dir is None:
                    output_dir = line.split(":", 1)[1].strip().strip("'\"")

    if not base_model:
        raise ValueError("Missing `model_name_or_path` in YAML.")
    if not output_dir:
        raise ValueError("Missing `output_dir` in YAML.")

    return str(base_model), str(output_dir)


def find_latest_checkpoint_dir(parent: str) -> Optional[str]:
    if not os.path.isdir(parent):
        return None

    candidates = [
        d for d in glob.glob(os.path.join(parent, "checkpoint-*")) if os.path.isdir(d)
    ]
    if not candidates:
        return None

    def step_num(path: str) -> int:
        m = re.search(r"checkpoint-(\d+)$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    candidates.sort(key=step_num)
    return candidates[-1]


def _dir_has_adapter(dir_path: str) -> bool:
    if not os.path.isdir(dir_path):
        return False
    if os.path.isfile(os.path.join(dir_path, "adapter_config.json")):
        return True
    # Fallback: accept presence of adapter_model.* alongside config in some setups
    adapter_bin = os.path.isfile(os.path.join(dir_path, "adapter_model.bin"))
    adapter_st = os.path.isfile(os.path.join(dir_path, "adapter_model.safetensors"))
    return adapter_bin or adapter_st


def pick_adapter_path(output_dir: str, explicit_adapter: Optional[str] = None) -> str:
    # 1) explicit
    if explicit_adapter:
        if os.path.isdir(explicit_adapter) and os.path.isfile(
            os.path.join(explicit_adapter, "adapter_config.json")
        ):
            return explicit_adapter
        raise FileNotFoundError(
            f"Provided adapter_path '{explicit_adapter}' is not a valid PEFT adapter directory."
        )

    # 2) output_dir direct save
    if _dir_has_adapter(output_dir):
        return output_dir

    # 3) pissa_converted (if any)
    pissa_converted = os.path.join(output_dir, "pissa_converted")
    if os.path.isdir(pissa_converted) and os.path.isfile(
        os.path.join(pissa_converted, "adapter_config.json")
    ):
        return pissa_converted

    # 4) last checkpoint under output_dir
    last_ckpt = find_latest_checkpoint_dir(output_dir)
    if last_ckpt and _dir_has_adapter(last_ckpt):
        return last_ckpt

    raise FileNotFoundError(
        f"Could not locate a LoRA adapter in '{output_dir}'."
        " Expected 'adapter_config.json' either directly in output_dir, in 'pissa_converted',"
        " or inside the latest 'checkpoint-*' directory."
    )


def list_checkpoint_dirs(output_dir: str) -> list[str]:
    if not os.path.isdir(output_dir):
        return []
    dirs = [d for d in glob.glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(d)]
    dirs.sort()
    return dirs


def clean_checkpoints(output_dir: str, dry_run: bool = False) -> list[str]:
    targets = list_checkpoint_dirs(output_dir)
    if dry_run:
        return targets
    for d in targets:
        try:
            shutil.rmtree(d)
        except Exception as e:
            print(f"[WARN] Failed to remove {d}: {e}")
    return targets


def load_base_model(model_path: str):
    # Prefer CausalLM; fall back to generic AutoModel if needed
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
        )
    except Exception:
        return AutoModel.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
        )


def merge_lora(base_model_path: str, adapter_path: str, save_dir: str, save_tokenizer: bool = True) -> None:
    os.makedirs(save_dir, exist_ok=True)

    model = load_base_model(base_model_path)
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(save_dir)

    if save_tokenizer:
        try:
            tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            tok.save_pretrained(save_dir)
        except Exception as e:
            print(f"[WARN] Failed to save tokenizer: {e}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model using a training YAML.")
    parser.add_argument("--yaml", required=True, help="Path to training YAML file.")
    parser.add_argument(
        "--base_model",
        default=None,
        help="Override base model path. If not provided, read from YAML 'model_name_or_path'.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override training output dir. If not provided, read from YAML 'output_dir'.",
    )
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Optional explicit adapter directory. Otherwise auto-detect in output_dir.",
    )
    parser.add_argument(
        "--save_subdir",
        default="merged",
        help="Subdirectory under output_dir to save the merged model (default: merged)",
    )
    parser.add_argument(
        "--no_save_tokenizer",
        action="store_true",
        help="Do not save tokenizer into the merged folder.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print resolved paths and exit without merging.",
    )
    parser.add_argument(
        "--clean_checkpoints",
        action="store_true",
        help="After merging, remove all 'checkpoint-*' subdirectories under output_dir.",
    )
    parser.add_argument(
        "--clean_only",
        action="store_true",
        help="Only clean all 'checkpoint-*' subdirectories under output_dir and exit.",
    )

    args = parser.parse_args()

    yaml_base_model, yaml_output_dir = read_yaml_keys(args.yaml)
    base_model = args.base_model or yaml_base_model
    output_dir = args.output_dir or yaml_output_dir

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"output_dir does not exist: {output_dir}")

    # Clean-only path
    if args.clean_only:
        cleaned = clean_checkpoints(output_dir, dry_run=args.dry_run)
        print(json.dumps({"cleaned_checkpoints": cleaned, "dry_run": args.dry_run}, ensure_ascii=False))
        return

    adapter_dir = pick_adapter_path(output_dir, args.adapter_path)
    save_dir = os.path.join(output_dir, args.save_subdir)

    info = {
        "base_model": base_model,
        "adapter_dir": adapter_dir,
        "save_dir": save_dir,
        "output_dir": output_dir,
    }
    print(json.dumps(info, ensure_ascii=False))

    if args.dry_run:
        return

    merge_lora(base_model, adapter_dir, save_dir, save_tokenizer=(not args.no_save_tokenizer))
    print(f"Merged model saved to: {save_dir}")

    if args.clean_checkpoints:
        cleaned = clean_checkpoints(output_dir, dry_run=False)
        print(json.dumps({"cleaned_checkpoints": cleaned}, ensure_ascii=False))


if __name__ == "__main__":
    main()


