"""DDR capture layout shared by the MoE recipe, streams and host manifest."""
import os


def enabled():
    return os.environ.get("OFLM_MOE_CAPTURE") == "1"


def layout(hidden, ff, topk, cores=8):
    fields, offset = {}, 0
    # Hidden streams still transfer a 4 KB element, even when FF is only 512.
    for name in ("up", "gate", "silu", "product"):
        fields[name] = {"offset": offset, "stride": max(4096, ff * 4), "dtype": "f32", "shape": [ff]}
        offset += (topk + 1) * fields[name]["stride"]
    for name in ("down", "accumulator"):
        fields[name] = {"offset": offset, "stride": hidden * 4, "dtype": "f32", "shape": [hidden]}
        offset += (topk + 1) * hidden * 4
    fields["down_input"] = {"offset": offset, "stride": cores * ff * 2, "dtype": "bf16", "shape": [cores, ff]}
    offset += (topk + 1) * cores * ff * 2
    fields["shared_gate"] = {"offset": offset, "stride": 256, "dtype": "f32", "shape": [1]}
    offset += cores * 256
    return {"version": 1,
            "capabilities": ["native_router", "native_moe_vectors", "native_expert_intermediates",
                             "native_bf16_down_input", "masked_contribution"],
            "bytes": (offset + 4095) // 4096 * 4096,
            "router_logits_offset": 1088, "control_offset": 2112, "fields": fields}
