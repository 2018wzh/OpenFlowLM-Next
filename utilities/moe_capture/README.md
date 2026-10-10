# Qwen3.6 MoE capture

This workflow records **OpenFlowLM's open NPU computation**, including its
native Q4NX packing and approximate vector arithmetic. It does not represent
the closed FastFlowLM engine or an unquantized Hugging Face implementation.
It uses all 40 layers, 256 routed experts and Top8, plus the shared expert.
Prefill, decode and teacher forcing use the same per-token NPU schedule.

## Build and run

Install the XRT SDK and set up a checkout of mlir-aie at **v1.4.2** using its
`utils/iron_setup.py`. That installs its matching Peano toolchain. The
Windows SDK's tools and the standalone runtime build also work natively on
Windows. Do not replace the installed NPU driver as part of this workflow.

Use the model package `Atomic-Germ/Qwen3.6-35B-A3B-NPU2`, with matching
`config.json`, `model.q4nx` and tokenizer files. No vision weights are needed.
Download these files with `huggingface_hub`; keep one copy of the text weights.

```sh
python utilities/moe_capture/download.py

python utilities/moe_capture/build.py --sdk "$XRT_ROOT" --iron "$IRON_ENV" \
  --model models/Qwen3.6-35B-A3B-NPU2

python src/open_qwen36/capture.py \
  --model models/Qwen3.6-35B-A3B-NPU2 \
  --kernels src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels_capture \
  --output captures/bank
```

`build.py --only runtime` or `--only kernels` performs an incremental build.
The source, binaries and captured datasets remain separate. Capture exports
do not produce toolchain comparison reports. Generated xclbins and datasets
are ignored by Git. Python dependencies for the capture frontend are NumPy
and `tokenizers`; the IRON environment has its own upstream dependencies.

Capture attention/router and MoE phases have independent program images and
hardware contexts. This keeps each main core's program within its 16 KiB
instruction memory budget; the runtime submits their dispatches serially.

The default cases are `He waited beside the river bank` and
`He waited inside the local bank`. Input text is encoded directly without a
chat template. Actual tokenizer character offsets locate the last `bank`;
all its overlapping tokens receive vectors in layers 0 and 3. The last such
token is the intervention point. The layer-3 baseline expert at rank 0 is
the microscope/target, and rank 7 is the selected-expert control. Ranks are
zero-based and do not imply any semantic specialization.

Baseline greedy generation produces up to eight tokens, ending at EOS. That
continuation is then fixed for baseline, target and control teacher forcing.
Each run resets all recurrent/convolution state and KV storage. Ablation
skips the target's MAC contribution at precisely one token/layer; native
Top8 IDs/probabilities/weights remain intact, other weights are not
renormalized, and the shared expert stays active. Later routing may change
as a consequence of the intervention.

## Custom cases and native interface

Pass `--cases cases.json` containing an array such as:

```json
[
  {"id":"example","text":"He waited beside the river bank","focus":"bank",
   "vector_layers":[0,3],"micro_layers":[3],"micro_rank":0,"ablate_layer":3,
   "max_tokens":8,"continuation":" watching the water."}
]
```

`continuation` is optional; otherwise baseline generation supplies it.
`capture_positions` and `target_position` can explicitly select zero-based
token positions. The intervention must be a selected prompt position, and
microscope layers must be selected vector layers; the ablation layer may be chosen independently.

The Python entrypoint creates a temporary native request and invokes
`open_qwen36_cli --capture-config request.json --capture-out OUTPUT`.
Native version-1 requests contain `cases` with compiled `ids`, token pieces
and character spans, positions/layers, and optional `continuation_ids`.
The driver loads weights once and processes all cases sequentially.
Capture requires a manifest with `moe_capture.version == 1`. Missing capture
kernels fail explicitly; there is no CPU or closed-engine fallback.

## Dataset

* `events.jsonl`: metadata, case/run boundaries, routes, selected MoE inputs
  and outputs, expert detail, logits, comparisons, and a final completion event.
* `tensors.bin`: a continuous little-endian tensor stream. Each descriptor
  specifies `name`, `dtype`, `shape`, `offset`, `length` and `source`.
* `summary.json`: decoded fixed continuations and paired probability/log
  likelihood differences, labelled `derived`.

Events identify the case/run, **input** token ID and zero-based position,
phase (`prefill`, `decode`, `teacher_force`), and layer. Logit events separately
identify the **next** token they predict. Distributions use only the real
vocabulary, excluding padded head columns.

Routes contain native logits, probabilities, dispatched IDs and weights for
every position/layer. Selected vectors include normalized input, residual
before/after, raw routed/shared down outputs, routed mixture before shared
closure, and the native shared sigmoid gate. Expert detail contains native
gate, up, SiLU(gate), product, rounded BF16 down input and down output.
The BF16 input is tapped from the values actually passed to table preparation
on core 0; every main core prepares that same full hidden vector.

```python
from capture_reader import CaptureDataset  # add src/open_qwen36 to sys.path

data = CaptureDataset("captures/bank")
data.require_complete()
for event in data.events():
    for descriptor in event.get("tensors", []):
        raw = data.tensor(descriptor)                 # BF16 stays uint16 bits
        floats = data.tensor(descriptor, as_float=True)
```

`weighted_contributions(route, output)` calculates **derived** float32
products with the effective ablation mask. These products can differ slightly
from the native split-BF16 accumulator's MACs; use `routed_mixture` for the
actual native sum. Do not label CPU-derived products as native intermediates.

Writing is synchronous with bounded stream/device buffers. Disk errors stop
execution; output without the normal `complete` event is incomplete. Existing
datasets are never overwritten. Use a new output directory for a new run.

Metadata records the projection quantization selected from the actual weight
container and the MoE execution precision. The current open recipe executes
routed and shared expert matrices as Q4_1; its existing weight packer converts
the package's Q8 shared matrices to that format. Captured values describe this
open execution, including that conversion, BF16 table inputs and approximate
split-BF16 MACs.

## Incremental tests

```sh
python utilities/moe_capture/test_capture.py

python utilities/moe_capture/acceptance.py --dataset captures/bank \
  --model models/Qwen3.6-35B-A3B-NPU2 \
  --kernels src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels_capture
```

For NPU acceptance, compare the same capture kernels with the runtime capture
hook off/on, using identical prompt IDs. Native routes and logits must remain
unchanged without ablation. Check both selected layer types and ensure only
the requested slot is masked at the intervention. Baseline generation and
baseline teacher forcing must agree on their shared prefix after reset.
The acceptance entrypoint uses the default greedy examples, performs these
incremental comparisons and removes its temporary logit dumps automatically.

实时通用 API、全部层采集及其他前端接入见 [API v1](API.md)。
