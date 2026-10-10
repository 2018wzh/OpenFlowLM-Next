"""Incremental NPU acceptance against a completed teaching dataset; no reports retained."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/open_qwen36"))
from capture_reader import CaptureDataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--kernels", type=Path, required=True)
    ap.add_argument("--exe", type=Path, default=ROOT / "src/open_qwen36/out/open_qwen36_cli.exe")
    a = ap.parse_args()
    data = CaptureDataset(a.dataset)
    data.require_complete()
    cases, routes, output, logits = {}, {}, {}, {}
    for e in data.events():
        kind = e["kind"]
        if kind == "case": cases[e["case"]] = e
        elif kind == "route": routes[(e["case"], e["run"], e["position"], e["layer"])] = e
        elif kind == "moe_output": output[(e["case"], e["run"], e["position"], e["layer"])] = e
        elif kind == "logits": logits.setdefault((e["case"], e["run"]), []).append(e)
        elif kind == "expert_detail":
            tensors = {t["name"]: t for t in e["tensors"]}
            assert set(tensors) == {"gate", "up", "silu", "product", "down_input", "down"}
            assert tensors["down_input"]["dtype"] == "bf16"
            for t in tensors.values():
                assert np.isfinite(data.tensor(t, as_float=True)).all()
    assert cases
    for case_id, case in cases.items():
        positions = {p for c, run, p, l in routes if c == case_id and run == "baseline"}
        for p in positions:
            assert {l for c, run, pos, l in routes if c == case_id and run == "baseline" and pos == p} == set(range(40))
        for key, e in routes.items():
            c, run, p, layer = key
            if c != case_id: continue
            base = routes[(c, "baseline", p, layer)]
            # Reset reproduces all native routes on the unablated run; up to and
            # including the intervention, paired inputs/route decisions match.
            if run == "baseline_generate" or (run in {"target", "control"} and (p, layer) <= (case["target_position"], case["ablate_layer"])):
                for t, bt in zip(e["tensors"], base["tensors"]):
                    np.testing.assert_array_equal(data.tensor(t), data.tensor(bt))
            at_target = p == case["target_position"] and layer == case["ablate_layer"]
            assert (e["ablate_slot"] >= 0) == (at_target and run in {"target", "control"})
            if at_target and run in {"target", "control"}:
                ids = data.tensor(next(t for t in e["tensors"] if t["name"] == "expert_ids"))
                base_ids = data.tensor(next(t for t in base["tensors"] if t["name"] == "expert_ids"))
                assert ids[e["ablate_slot"]] == base_ids[0 if run == "target" else 7]
        for layer in case["vector_layers"]:
            assert (case_id, "baseline", case["target_position"], layer) in output
        for run in ("baseline", "target", "control"):
            for p in case["capture_positions"]:
                for layer in case["vector_layers"]:
                    key = (case_id, run, p, layer)
                    out = output[key]
                    tensors = {t["name"]: t for t in out["tensors"]}
                    if run in {"target", "control"} and (p, layer) == (case["target_position"], case["ablate_layer"]):
                        original = output[(case_id, "baseline", p, layer)]
                        # The intervention only masks a contribution: each raw
                        # expert and the shared branch still execute unchanged.
                        for t, bt in zip(out["tensors"], original["tensors"]):
                            if t["name"] in {"expert_output", "shared_output", "shared_gate"}:
                                np.testing.assert_array_equal(data.tensor(t), data.tensor(bt))
                    mix = data.tensor(tensors["routed_mixture"])
                    total = sum(t["value"] for t in data.weighted_contributions(routes[key], out))
                    np.testing.assert_allclose(mix, total, rtol=2e-4, atol=2e-4)
        normal = logits[(case_id, "baseline_generate")]
        forced = logits[(case_id, "baseline")]
        for x, y in zip(normal, forced):
            assert x["next_token_id"] == y["next_token_id"]
            np.testing.assert_array_equal(data.tensor(x["tensors"][0]), data.tensor(y["tensors"][0]))
    print("PASS: all-layer routes, both layer types, native expert detail, reset and ablation")

    # One complete-model invocation checks the runtime hook off against hook on.
    # Temporary logit dumps are removed when the subprocess comparison finishes.
    case_id, case = next(iter(cases.items()))
    baseline = logits[(case_id, "baseline_generate")]
    env = os.environ.copy()
    env["OFLM_OPEN_ROUTE_SENTINEL"] = "1"
    env.pop("OFLM_ROUTE_CHECK", None)
    if os.name == "nt":
        env["PATH"] = str(Path(env.get("SystemRoot", "C:/Windows")) / "System32/AMD") + os.pathsep + env.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix="oflm-capture-test-") as tmp:
        prefix = Path(tmp) / "logits"
        subprocess.run([str(a.exe.resolve()), "--model", str(a.model.resolve()), "--kernels", str(a.kernels.resolve()),
                        "--ids", ",".join(map(str, case["input_ids"])), "--max-tokens", str(len(baseline)),
                        "--max-ctx", "128", "--decode-route", "npu", "--pmode", "none", "--dump-logits", str(prefix)],
                       env=env, check=True)
        for i, e in enumerate(baseline):
            raw = np.fromfile(f"{prefix}_t{i}.bin", dtype="<f4")
            captured = data.tensor(e["tensors"][0])
            np.testing.assert_array_equal(raw[:len(captured)], captured)
    print("PASS: native logits unchanged with runtime capture hook off/on")


if __name__ == "__main__":
    main()
