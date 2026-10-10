"""Incremental tests of the exported data interface and tokenizer alignment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/open_qwen36"))
from capture import prepare_cases
from capture_reader import CaptureDataset


class CaptureTests(unittest.TestCase):
    def test_focus_offsets_and_fixed_continuation(self):
        tk = Tokenizer(models.WordLevel({"[UNK]": 0, "bank": 1, "river": 2, "next": 3}, unk_token="[UNK]"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        c = prepare_cases([{"id": "sample", "text": "bank river bank", "focus": "bank", "continuation": "next"}], tk)
        self.assertEqual(c["cases"][0]["target_position"], 2)
        self.assertEqual(c["cases"][0]["capture_positions"], [2])
        self.assertEqual(c["cases"][0]["continuation_ids"], [3])
        with self.assertRaises(ValueError):
            prepare_cases([{"id": "x", "text": "river", "focus": "bank"}], tk)

    def test_bf16_bits_and_incomplete_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "tensors.bin").write_bytes(np.array([0x3f80, 0xc020], dtype="<u2").tobytes())
            (p / "events.jsonl").write_text('{"kind":"metadata"}\n', encoding="utf-8")
            reader = CaptureDataset(p)
            descriptor = {"dtype": "bf16", "shape": [2], "offset": 0, "length": 4}
            np.testing.assert_array_equal(reader.tensor(descriptor), [0x3f80, 0xc020])
            np.testing.assert_array_equal(reader.tensor(descriptor, as_float=True), [1, -2.5])
            with self.assertRaises(ValueError):
                reader.require_complete()
            with (p / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": "complete", "tensor_bytes": 4}) + '\n')
            reader.require_complete()

    def test_derived_ablation_keeps_original_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            values = np.array([.75, .25, 2, 4], dtype="<f4")
            (p / "tensors.bin").write_bytes(values.tobytes())
            reader = CaptureDataset(p)
            key = {"case": "x", "run": "target", "position": 2, "layer": 3}
            weights = {"name": "expert_weights", "dtype": "f32", "shape": [2], "offset": 0, "length": 8}
            route = key | {"ablate_slot": 0, "tensors": [weights]}
            output = key | {"tensors": [{"name": "expert_output", "expert_id": i, "slot": i,
                          "dtype": "f32", "shape": [1], "offset": 8 + 4 * i, "length": 4} for i in range(2)]}
            contributions = reader.weighted_contributions(route, output)
            np.testing.assert_array_equal(contributions[0]["value"], [0])
            np.testing.assert_array_equal(contributions[1]["value"], [1])
            np.testing.assert_array_equal(reader.tensor(weights), [.75, .25])


if __name__ == "__main__":
    unittest.main()
