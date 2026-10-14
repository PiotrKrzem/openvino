#!/usr/bin/env python3
"""
convert.py -- MiniCPM5-1B -> OpenVINO IR conversion (Phase 1).

MiniCPM is a decoder-only causal language model. The supported and
reproducible conversion path for OpenVINO is optimum-intel, which exports the
HuggingFace checkpoint to a *stateful* OpenVINO IR (KV-cache kept as model
state). This is the same path used by OpenVINO GenAI.

Usage:
    python convert.py --model-id openbmb/MiniCPM-1B-sft-bf16 \
                      --output ./ov_minicpm --weight-format fp16

Notes:
  * MiniCPM uses `trust_remote_code=True` (custom modeling file on the Hub).
  * No custom CUDA kernels are required for export; export runs CPU-only.
  * `--weight-format` selects the IR precision (fp32 / fp16 / int8 / int4).
"""
import argparse
import sys
from pathlib import Path


def convert(model_id: str, output: str, weight_format: str, trust_remote_code: bool) -> int:
    try:
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        print(
            "[ERROR] optimum-intel is required. Install with:\n"
            "    pip install 'optimum-intel[openvino]' transformers\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_kwargs = {
        "export": True,
        "trust_remote_code": trust_remote_code,
    }
    # Weight compression to int8/int4 is driven by an explicit weight-only
    # quantization config (optimum-intel >= 1.16).
    if weight_format in ("int8", "int4"):
        from optimum.intel import OVWeightQuantizationConfig

        export_kwargs["quantization_config"] = OVWeightQuantizationConfig(
            bits=8 if weight_format == "int8" else 4
        )

    print(f"[INFO] Exporting {model_id} -> {out_dir} (weight_format={weight_format})")
    ov_model = OVModelForCausalLM.from_pretrained(model_id, **export_kwargs)

    if weight_format == "fp16":
        ov_model.half()

    ov_model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code).save_pretrained(out_dir)

    xml = out_dir / "openvino_model.xml"
    if not xml.exists():
        print(f"[ERROR] Expected IR not found at {xml}", file=sys.stderr)
        return 1
    print(f"[SUCCESS] IR written to {xml}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="openbmb/MiniCPM-1B-sft-bf16")
    parser.add_argument("--output", default="./ov_minicpm")
    parser.add_argument(
        "--weight-format",
        default="fp16",
        choices=["fp32", "fp16", "int8", "int4"],
    )
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()
    sys.exit(convert(args.model_id, args.output, args.weight_format, args.trust_remote_code))
