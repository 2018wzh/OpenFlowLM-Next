---
name: moe-capture
description: Build the dedicated OpenFlowLM Qwen3.6 per-token NPU capture kernels and run paired expert ablations.
---

# MoE capture

Read `utilities/moe_capture/README.md` from the repository root. Build only
the standalone runtime and six Qwen3.6 capture kernel sets through
`utilities/moe_capture/build.py`. Use mlir-aie v1.4.2 and its pinned Peano;
an XRT SDK provides xclbinutil and aiebu-asm. Windows builds are native.

`--capture` selects separate layer build directories and a capture-capable
manifest. Capture phase 0 and phase 1 have independent program images and
hardware contexts, with each main worker containing only its phase. This
leaves program-memory room for taps. Edit `gen_kernels.py`, not its
generated C++ files. The shared DDR layout is `recipes/moe_capture.py`.

Capture uses only NPU per-token dispatches. Collect the router record after
the existing freshness handling, and preserve its IDs/probabilities/weights.
Native scratch is drained before reuse; never substitute a host replica for
missing data. The down-input tap must share the actual BF16 conversion used
by GEMV table preparation. Ablate only the specified weighted contribution.

Keep dataset metadata free of machine paths. Data without a normal completion
event is incomplete. Retain final datasets and reusable build products;
remove obsolete temporary artifacts. Run only tests relevant to changes.
