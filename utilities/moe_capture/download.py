"""Download the text-only files of the matched Qwen3.6 OpenFlowLM package."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/Qwen3.6-35B-A3B-NPU2")
    args = parser.parse_args()
    snapshot_download("Atomic-Germ/Qwen3.6-35B-A3B-NPU2", local_dir=args.output,
                      allow_patterns=["model.q4nx", "config.json", "tokenizer.json",
                                      "tokenizer_config.json", "chat_template.jinja"], max_workers=2)
