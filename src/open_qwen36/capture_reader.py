"""Streaming event reader and zero-copy NumPy tensor views for MoE captures."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DTYPES = {"f32": np.dtype("<f4"), "i32": np.dtype("<i4"), "bf16": np.dtype("<u2")}


class CaptureDataset:
    def __init__(self, directory):
        self.directory = Path(directory)

    def events(self):
        with (self.directory / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)

    def require_complete(self):
        last = None
        for last in self.events():
            pass
        if last is None or last.get("kind") != "complete":
            raise ValueError("capture is incomplete: no normal completion event")
        if last["tensor_bytes"] != (self.directory / "tensors.bin").stat().st_size:
            raise ValueError("capture tensor file does not match its completion record")

    def tensor(self, descriptor, *, as_float=False):
        dtype = DTYPES[descriptor["dtype"]]
        shape = tuple(descriptor["shape"])
        count = int(np.prod(shape, dtype=np.int64))
        offset, length = descriptor["offset"], descriptor["length"]
        if count <= 0 or offset < 0 or count * dtype.itemsize != length:
            raise ValueError("invalid tensor dimensions or length")
        if offset + length > (self.directory / "tensors.bin").stat().st_size:
            raise ValueError("tensor data is truncated")
        value = np.memmap(self.directory / "tensors.bin", mode="r", dtype=dtype,
                          offset=offset, shape=shape)
        if descriptor["dtype"] == "bf16" and as_float:
            return (value.astype(np.uint32) << 16).view(np.float32)
        return value

    def weighted_contributions(self, route, output):
        """Derived float32 products, distinct from the native accumulator's MACs."""
        keys = ("case", "run", "position", "layer")
        if any(route[k] != output[k] for k in keys):
            raise ValueError("route and output refer to different computation points")
        weights = self.tensor(next(t for t in route["tensors"] if t["name"] == "expert_weights"))
        contributions = []
        for t in output["tensors"]:
            if t["name"] == "expert_output":
                slot = t["slot"]
                value = self.tensor(t) * weights[slot]
                if route.get("ablate_slot", -1) == slot:
                    value = np.zeros_like(value)
                contributions.append({"expert_id": t["expert_id"], "slot": slot,
                                      "source": "derived", "value": value})
        return contributions
