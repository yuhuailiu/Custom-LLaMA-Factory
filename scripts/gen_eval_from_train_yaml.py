#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import yaml  # type: ignore


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_training_yaml(yaml_path: Path) -> Tuple[str, str, Optional[str]]:
    """Parse training YAML to extract model_name_or_path, output_dir, finetuning_type."""
    data = yaml.safe_load(_read_text(yaml_path))
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {yaml_path} does not contain a top-level mapping.")
    base_model = data.get("model_name_or_path")
    output_dir = data.get("output_dir")
    finetuning_type = data.get("finetuning_type")

    if not base_model:
        raise ValueError("Missing `model_name_or_path` in training YAML.")
    if not output_dir:
        raise ValueError("Missing `output_dir` in training YAML.")

    return str(base_model), str(output_dir), (str(finetuning_type) if finetuning_type else None)


def resolve_eval_inputs(output_dir: str, finetuning_type: Optional[str]) -> Tuple[str, str, str]:
    """Return (mode, model_path_for_eval, eval_output_dir).

    - mode: "lora" or "full" detected from finetuning_type
    - model_path_for_eval: for lora -> output_dir/merged, for full -> output_dir
    - eval_output_dir: /data2/lyh/eval_results/<basename>_<mode>
    """
    lora_like = {"lora", "qlora", "pissa"}
    mode = "lora" if (finetuning_type and str(finetuning_type).lower() in lora_like) else "full"

    train_out = Path(output_dir)
    if mode == "lora":
        model_path = train_out / "merged"
    else:
        model_path = train_out

    # Construct eval output dir under a common root
    eval_root = Path("/data2/lyh/eval_results")
    eval_out = eval_root / f"{train_out.name}_{mode}"

    return mode, model_path.as_posix(), eval_out.as_posix()


def load_eval_template(template_path: Path) -> str:
    return _read_text(template_path)


def update_eval_yaml_yaml(template_text: str, model_path: str, eval_output_dir: str, mode: str) -> str:
    """Parse and dump to preserve YAML structure using PyYAML."""
    data = yaml.safe_load(template_text)
    if not isinstance(data, dict):
        raise ValueError("Eval template is not a mapping at top level.")
    data["model_name_or_path"] = model_path
    data["output_dir"] = eval_output_dir
    data["finetuning_type"] = "lora" if mode == "lora" else "full"
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an evaluation YAML from a training YAML.")
    parser.add_argument("--train_yaml", required=True, help="Path to the training YAML.")
    parser.add_argument(
        "--eval_template",
        default="lyh_yamls/eval/EVALUATION_TEMPLATE.yaml",
        help="Path to the evaluation template YAML.",
    )
    parser.add_argument(
        "--save_dir",
        default="lyh_yamls/eval",
        help="Directory to write the generated evaluation YAML.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional explicit filename for the generated YAML. If omitted, a name is derived from training output_dir.",
    )
    parser.add_argument(
        "--fail_if_merged_missing",
        action="store_true",
        help="If set and finetuning_type is LoRA-like, fail when output_dir/merged does not exist.",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run evaluation immediately after generating the eval YAML.",
    )
    parser.add_argument(
        "--python_executable",
        default=sys.executable,
        help="Python executable to launch the eval module (default: current interpreter).",
    )
    parser.add_argument(
        "--eval_module",
        default="llamafactory.cli",
        help="Python module to run for evaluation (default: llamafactory.cli).",
    )
    parser.add_argument(
        "--run_eval_inproc",
        action="store_true",
        help="Run evaluation in-process by importing LLaMA-Factory (no subprocess). Only supported for the 'eval' subsystem (requires 'task' keys).",
    )

    args = parser.parse_args()

    train_yaml_path = Path(args.train_yaml)
    eval_template_path = Path(args.eval_template)
    save_dir = Path(args.save_dir)

    base_model_path, train_output_dir, finetuning_type = parse_training_yaml(train_yaml_path)
    mode, model_for_eval, eval_output_dir = resolve_eval_inputs(train_output_dir, finetuning_type)

    if mode == "lora":
        merged_dir = Path(model_for_eval)
        if not merged_dir.is_dir():
            msg = (
                f"Merged model directory not found: {merged_dir}\n"
                f"Please merge LoRA first, e.g.:\n"
                f"python scripts/merge_lora_from_yaml.py --yaml {train_yaml_path}"
            )
            if args.fail_if_merged_missing:
                raise FileNotFoundError(msg)
            else:
                print(f"[WARN] {msg}")

    template_text = load_eval_template(eval_template_path)
    updated_yaml = update_eval_yaml_yaml(template_text, model_for_eval, eval_output_dir, mode)

    # Derive output filename
    if args.filename:
        out_name = args.filename
    else:
        bn = Path(train_output_dir).name
        out_name = f"EVAL_{bn}_{mode}.yaml"

    out_path = (save_dir / out_name).resolve()
    _write_text(out_path, updated_yaml)

    summary = {
        "train_yaml": str(train_yaml_path.resolve()),
        "resolved_base_model": base_model_path,
        "train_output_dir": train_output_dir,
        "mode": mode,
        "model_for_eval": model_for_eval,
        "eval_output_dir": eval_output_dir,
        "generated_eval_yaml": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.run_eval:
        cmd = [
            args.python_executable,
            "-m",
            args.eval_module,
            "eval",
            str(out_path),
        ]
        print(json.dumps({"running": "eval", "cmd": " ".join(cmd)}, ensure_ascii=False))
        subprocess.run(cmd, check=True)

    if args.run_eval_inproc:
        # In-process path supports the standalone eval subsystem (Evaluator), which expects 'task' fields.
        config_dict = yaml.safe_load(updated_yaml)
        if not isinstance(config_dict, dict):
            raise ValueError("Generated eval YAML is not a mapping.")
        if "task" not in config_dict:
            raise RuntimeError(
                "In-process eval requires the 'evaluation' subsystem config (with 'task', 'task_dir', etc.). "
                "Your template appears to be a training-eval YAML (do_eval with datasets). "
                "Please use --run_eval to launch via the CLI, or provide an eval YAML with 'task'."
            )
        # Lazy import to avoid heavy deps unless requested
        from llamafactory.eval.evaluator import Evaluator  # type: ignore
        Evaluator(args=config_dict).eval()


if __name__ == "__main__":
    main()


