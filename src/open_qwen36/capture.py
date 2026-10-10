"""Record the two bank examples, or cases from a JSON file, on the open NPU engine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from tokenizers import Tokenizer

DEFAULT_CASES = [
    {"id": "river_bank", "text": "He waited beside the river bank", "focus": "bank"},
    {"id": "local_bank", "text": "He waited inside the local bank", "focus": "bank"},
]


def prepare_cases(cases, tokenizer, max_tokens=8):
    compiled, seen = [], set()
    for case in cases:
        case_id, text = case["id"], case["text"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case ids must be nonempty and unique")
        seen.add(case_id)
        encoding = tokenizer.encode(text, add_special_tokens=False)
        if not encoding.ids:
            raise ValueError(f"{case_id}: empty prompt")
        focus = case.get("focus")
        if focus is not None and (not isinstance(focus, str) or not focus or focus not in text):
            raise ValueError(f"{case_id}: focus text not in prompt")
        start = text.rindex(focus) if focus else 0
        end = start + len(focus) if focus else len(text)
        positions = [i for i, (a, b) in enumerate(encoding.offsets) if a < end and b > start] if focus else [len(encoding.ids)-1]
        if not positions:
            raise ValueError(f"{case_id}: tokenizer does not map the focus to an input token")
        value = dict(case)
        value.update(ids=encoding.ids,
                     tokens=[{"id": i, "piece": p, "span": list(o)}
                             for i, p, o in zip(encoding.ids, encoding.tokens, encoding.offsets)],
                     capture_positions=case.get("capture_positions", positions),
                     target_position=case.get("target_position", positions[-1]),
                     vector_layers=case.get("vector_layers", [0, 3]),
                     micro_layers=case.get("micro_layers", [3]), micro_rank=case.get("micro_rank", 0),
                     ablate_layer=case.get("ablate_layer", 3), max_tokens=case.get("max_tokens", max_tokens))
        if value["capture_positions"] == "all":
            value["capture_positions"] = list(range(len(encoding.ids)))
        if value["vector_layers"] == "all":
            value["vector_layers"] = list(range(40))
        if value["target_position"] not in value["capture_positions"]:
            raise ValueError(f"{case_id}: target position must be selected for detailed capture")
        if value["micro_layers"] == "all":
            value["micro_layers"] = list(value["vector_layers"])
        for field in ("capture_positions", "vector_layers", "micro_layers"):
            if not isinstance(value[field], list) or any(type(i) is not int for i in value[field]) or len(set(value[field])) != len(value[field]):
                raise ValueError(f"{case_id}: {field} must be an array of unique integers")
        for field in ("target_position", "micro_rank", "ablate_layer"):
            if type(value[field]) is not int:
                raise ValueError(f"{case_id}: {field} must be an integer")
        if any(type(l) is not int or l not in value["vector_layers"] for l in value["micro_layers"]) or not 0 <= value["ablate_layer"] < 40:
            raise ValueError(f"{case_id}: microscope must use a selected layer; ablation layer outside model")
        if not 0 <= value["micro_rank"] < 8:
            raise ValueError(f"{case_id}: micro_rank must be in [0, 7]")
        if type(value["max_tokens"]) is not int or value["max_tokens"] < 1 or any(type(l) is not int or not 0 <= l < 40 for l in value["vector_layers"]):
            raise ValueError(f"{case_id}: invalid generation length or layer")
        if not 0 <= value["target_position"] < len(encoding.ids):
            raise ValueError(f"{case_id}: target position outside prompt")
        if any(not 0 <= p < len(encoding.ids) for p in value["capture_positions"]):
            raise ValueError(f"{case_id}: capture position outside prompt")
        if "continuation" in case:
            value["continuation_ids"] = tokenizer.encode(case["continuation"], add_special_tokens=False).ids
            if not value["continuation_ids"]:
                raise ValueError(f"{case_id}: empty continuation")
        # Only explicit execution fields cross into the native driver; no host paths.
        ranks = case.get("ablation_ranks", [0, 7])
        if not isinstance(ranks, list) or len(set(ranks)) != len(ranks) or any(type(r) is not int or not 0 <= r < 8 for r in ranks):
            raise ValueError(f"{case_id}: ablation_ranks must be unique ranks in [0, 7]")
        value["ablation_ranks"] = ranks
        compiled.append({k: value[k] for k in ("id", "text", "ids", "tokens", "capture_positions",
                          "target_position", "vector_layers", "micro_layers", "micro_rank", "ablate_layer", "ablation_ranks", "max_tokens")}
                        | ({"continuation_ids": value["continuation_ids"]} if "continuation_ids" in value else {}))
    if not compiled:
        raise ValueError("at least one case is required")
    eos = [tokenizer.token_to_id(s) for s in ("<|im_end|>", "<|endoftext|>")]
    return {"version": 1, "eos_ids": [i for i in eos if i is not None], "cases": compiled}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--kernels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", type=Path, help="JSON array of {id,text,focus,continuation?} cases")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--max-ctx", type=int, default=128)
    ap.add_argument("--exe", type=Path, default=Path(__file__).parent / "out/open_qwen36_cli.exe")
    args = ap.parse_args()
    if args.output.exists() and any((args.output / f).exists() for f in ("events.jsonl", "tensors.bin")):
        ap.error("output already contains a dataset; choose a new output directory")
    if args.max_ctx < 1:
        ap.error("--max-ctx must be positive")
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    cases = json.loads(args.cases.read_text(encoding="utf-8")) if args.cases else DEFAULT_CASES
    config = prepare_cases(cases, tokenizer, args.max_tokens)
    for case in config["cases"]:
        needed = len(case["ids"]) + max(case["max_tokens"], len(case.get("continuation_ids", [])))
        if needed > args.max_ctx:
            ap.error("case exceeds --max-ctx")
    with tempfile.TemporaryDirectory(prefix="oflm-capture-") as temporary:
        request = Path(temporary) / "config.json"
        request.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        env = os.environ.copy()
        if os.name == "nt":
            runtime = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/AMD"
            env["PATH"] = str(runtime) + os.pathsep + env.get("PATH", "")
        subprocess.run([str(args.exe.resolve()), "--model", str(args.model.resolve()),
                        "--kernels", str(args.kernels.resolve()), "--max-ctx", str(args.max_ctx),
                        "--capture-config", str(request), "--capture-out", str(args.output.resolve()),
                        "--pmode", "none"], check=True, env=env)
    from capture_reader import CaptureDataset
    dataset = CaptureDataset(args.output)
    dataset.require_complete()
    summary = {"version": 1, "source": "derived", "cases": {}}
    for event in dataset.events():
        if event["kind"] == "continuation":
            summary["cases"].setdefault(event["case"], {}).update(
                continuation_ids=event["ids"], continuation_text=tokenizer.decode(event["ids"]))
        elif event["kind"] == "comparison":
            summary["cases"].setdefault(event["case"], {})["scores"] = event["scores"]
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
