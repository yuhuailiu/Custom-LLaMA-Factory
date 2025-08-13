### View training_args.bin

This guide shows how to inspect a binary `training_args.bin` (Transformers TrainingArguments) and optionally export it to JSON.

Replace the path with your actual file if different:

```
/data2/lyh/output_models/qwen_7b_lora256_pretrain1M_2klen_32Seq/training_args.bin
```

#### Print to terminal

```bash
python - <<'PY'
import json, torch, transformers  # ensure class is importable for deserialization
p = "/data2/lyh/output_models/qwen_7b_lora256_pretrain1M_2klen_32Seq/training_args.bin"
args = torch.load(p, map_location="cpu")
print(json.dumps(args.to_dict(), indent=2, ensure_ascii=False))
PY
```

#### Export to JSON file

```bash
python - <<'PY'
import json, torch, transformers, pathlib
p = pathlib.Path("/data2/lyh/output_models/qwen_7b_lora256_pretrain1M_2klen_32Seq/training_args.bin")
args = torch.load(p.as_posix(), map_location="cpu").to_dict()
out = p.with_suffix(".json")
out.write_text(json.dumps(args, indent=2, ensure_ascii=False))
print(f"Wrote {out}")
PY
```

#### Notes

- If you encounter deserialization errors, ensure a compatible `transformers` version is installed and that `import transformers` happens before `torch.load` (both scripts above already do this).
- The JSON export will be written next to the `.bin` file as `training_args.json`.


