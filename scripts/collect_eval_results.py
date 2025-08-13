#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


HF_JSON_CANDIDATES = [
    "eval_results.json",
    "all_results.json",
]


def is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except Exception:
        return False


def parse_mode_from_name(name: str) -> str:
    suffix = name.lower()
    if suffix.endswith("_lora"):
        return "lora"
    if suffix.endswith("_full"):
        return "full"
    return "unknown"


def parse_hf_metrics(dir_path: Path) -> Dict[str, float]:
    # Prefer known filenames; else pick JSON with max eval_* keys
    metrics = {}
    for fname in HF_JSON_CANDIDATES:
        p = dir_path / fname
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    metrics.update({k: v for k, v in data.items() if isinstance(v, (int, float))})
                    return metrics
            except Exception:
                pass

    best = None
    best_count = -1
    for p in dir_path.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            eval_items = {k: v for k, v in data.items() if k.startswith("eval_") and isinstance(v, (int, float))}
            if len(eval_items) > best_count:
                best = eval_items
                best_count = len(eval_items)

    if best is not None:
        metrics.update(best)
    return metrics


def parse_evaluator_log(dir_path: Path) -> Dict[str, float]:
    # Parse results.log lines like: "         Average: 93.21"
    p = dir_path / "results.log"
    if not p.is_file():
        return {}
    out: Dict[str, float] = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^\s*([^:]+):\s*([-+]?\d+(?:\.\d+)?)\s*$", line)
        if m:
            key = m.group(1).strip()
            try:
                out[f"score_{key}"] = float(m.group(2))
            except Exception:
                pass
    return out


def collect_one(dir_path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {}
    row["run"] = dir_path.name
    row["path"] = str(dir_path)
    row["mode"] = parse_mode_from_name(dir_path.name)
    try:
        row["mtime"] = int(dir_path.stat().st_mtime)
    except Exception:
        row["mtime"] = ""

    # Try HF metrics first
    hf = parse_hf_metrics(dir_path)
    if hf:
        row["type"] = "hf"
        row.update(hf)
    # Also parse evaluator log if present (may coexist)
    ev = parse_evaluator_log(dir_path)
    if ev:
        row.setdefault("type", "evaluator")
        row.update(ev)
    return row


def collect_all(root: Path, recursive: bool) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not is_dir(root):
        return rows
    dirs: List[Path] = []
    if recursive:
        for p, dnames, _ in os.walk(root):
            for d in dnames:
                dirs.append(Path(p) / d)
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]

    for d in sorted(dirs):
        rows.append(collect_one(d))
    return rows


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> Tuple[int, List[str]]:
    # Determine columns: stable first columns + sorted metric columns
    base_cols = ["run", "mode", "type", "mtime", "path"]
    metric_keys = set()
    for r in rows:
        metric_keys.update(k for k in r.keys() if k not in base_cols)
    # Ensure base cols order and then sorted remaining
    cols = base_cols + sorted([k for k in metric_keys if k not in base_cols])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return len(rows), cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect evaluation results and export to CSV.")
    parser.add_argument(
        "--root",
        default="/data2/lyh/eval_results",
        help="Root directory containing evaluation result folders.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when collecting.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <root>/summary.csv",
    )

    args = parser.parse_args()
    root = Path(args.root)
    out_csv = Path(args.output) if args.output else (root / "summary.csv")

    rows = collect_all(root, recursive=args.recursive)
    count, cols = write_csv(rows, out_csv)
    print(json.dumps({"count": count, "csv": str(out_csv), "columns": cols}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


