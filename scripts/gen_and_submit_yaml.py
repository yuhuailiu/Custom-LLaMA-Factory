#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Tuple

import yaml


DEFAULT_TRAIN_TEMPLATE = \
    "/data2/lyh/Custom-LLaMA-Factory/lyh_yamls/train/stage1_pretrain/qwen_7b_pretrain_with_ts.yaml"
DEFAULT_EVAL_TEMPLATE = \
    "/data2/lyh/Custom-LLaMA-Factory/lyh_yamls/eval/eval_test.yaml"
SCHEDULER_CLI = \
    "/data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py"


def parse_kv_overrides(pairs: List[str]) -> Dict[str, Any]:
    def parse_value(raw: str) -> Any:
        # Try to coerce to python types (bool/int/float/list/...) if obvious
        raw_strip = raw.strip()
        # explicit null
        if raw_strip.lower() in {"null", "none"}:
            return None
        # booleans
        if raw_strip.lower() in {"true", "false"}:
            return raw_strip.lower() == "true"
        # try int
        try:
            if raw_strip.isdigit() or (raw_strip.startswith("-") and raw_strip[1:].isdigit()):
                return int(raw_strip)
        except Exception:
            pass
        # try float
        try:
            if any(ch in raw_strip for ch in [".", "e", "E"]):
                return float(raw_strip)
        except Exception:
            pass
        # list or dict via yaml
        try:
            v = yaml.safe_load(raw_strip)
            # keep only if it parsed into non-scalar collection
            if isinstance(v, (list, dict)):
                return v
        except Exception:
            pass
        # default: raw string
        return raw

    result: Dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}', expected key=value")
        key, val = item.split("=", 1)
        key = key.strip()
        val_parsed = parse_value(val)
        result[key] = val_parsed
    return result


def set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Dict[str, Any] = config
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        set_dotted(cfg, k, v)
    return cfg


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML is not a mapping: {path}")
    return data


def dump_yaml(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def derive_eval_from_train(train_cfg: Dict[str, Any], eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill eval config defaults from train config for corresponding LoRA eval.

    - model_name_or_path: same as train unless already set
    - adapter_name_or_path: train.output_dir
    - output_dir: default to train.output_dir + "/eval" if not set
    - Ensure do_eval true, do_train false
    """
    out = dict(eval_cfg)

    train_model = train_cfg.get("model_name_or_path")
    train_out = train_cfg.get("output_dir")
    if train_model and "model_name_or_path" not in out:
        out["model_name_or_path"] = train_model

    if train_out:
        out["adapter_name_or_path"] = train_out
        out.setdefault("output_dir", os.path.join(train_out, "eval"))

    # make sure these flags are consistent
    out["do_train"] = False
    out["do_eval"] = True

    # keep finetuning_type from template (usually lora is OK)
    return out


def make_timestamped_name(prefix: str, suffix: str = "", ext: str = ".yaml") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{prefix}_{suffix}_{ts}{ext}"
    return f"{prefix}_{ts}{ext}"


def submit_to_scheduler(task: str) -> None:
    # task is already like: train:/abs/path.yaml or eval:/abs/path.yaml
    cmd = [sys.executable, SCHEDULER_CLI, "submit", task]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate train/eval YAMLs from templates with overrides and submit to scheduler.")
    parser.add_argument("--train_template", default=DEFAULT_TRAIN_TEMPLATE)
    parser.add_argument("--eval_template", default=DEFAULT_EVAL_TEMPLATE)
    parser.add_argument("--set", dest="train_overrides", nargs="*", default=[], help="Overrides for train YAML as key=value. Support dotted keys.")
    parser.add_argument("--eval-set", dest="eval_overrides", nargs="*", default=[], help="Overrides for eval YAML as key=value. Support dotted keys.")
    parser.add_argument("--write_dir", default=None, help="Optional base dir to write generated YAMLs. Defaults: train to its output_dir; eval to its output_dir")
    parser.add_argument("--no_submit", action="store_true", help="Only generate YAMLs, do not submit to scheduler.")

    args = parser.parse_args()

    train_tpl = load_yaml(args.train_template)
    eval_tpl = load_yaml(args.eval_template)

    train_overrides = parse_kv_overrides(args.train_overrides)
    eval_overrides = parse_kv_overrides(args.eval_overrides)

    train_cfg = apply_overrides(dict(train_tpl), train_overrides)
    # Ensure required keys exist in train (output_dir is important for filenames)
    train_out_dir = train_cfg.get("output_dir")
    if not train_out_dir:
        raise ValueError("Train config must have output_dir (either in template or via overrides)")

    # Derive eval from train defaults, then apply explicit eval overrides
    eval_cfg = derive_eval_from_train(train_cfg, dict(eval_tpl))
    eval_cfg = apply_overrides(eval_cfg, eval_overrides)

    # Resolve write targets
    if args.write_dir:
        base_dir = os.path.abspath(args.write_dir)
        os.makedirs(base_dir, exist_ok=True)
        train_yaml_path = os.path.join(base_dir, make_timestamped_name("train"))
        eval_yaml_path = os.path.join(base_dir, make_timestamped_name("eval"))
    else:
        train_yaml_path = os.path.join(os.path.abspath(train_out_dir), make_timestamped_name("train"))
        eval_out_dir = os.path.abspath(eval_cfg.get("output_dir", os.path.join(train_out_dir, "eval")))
        os.makedirs(eval_out_dir, exist_ok=True)
        eval_yaml_path = os.path.join(eval_out_dir, make_timestamped_name("eval"))

    # Dump YAMLs
    dump_yaml(train_yaml_path, train_cfg)
    dump_yaml(eval_yaml_path, eval_cfg)

    print("Generated:")
    print("  train:", train_yaml_path)
    print("  eval :", eval_yaml_path)

    if args.no_submit:
        return

    # Submit to scheduler: train first, then eval
    submit_to_scheduler(f"train:{train_yaml_path}")
    submit_to_scheduler(f"eval:{eval_yaml_path}")
    print("Submitted to scheduler queue (train then eval).")


if __name__ == "__main__":
    main()


