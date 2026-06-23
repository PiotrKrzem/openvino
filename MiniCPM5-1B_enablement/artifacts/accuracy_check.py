#!/usr/bin/env python3
"""
accuracy_check.py -- MiniCPM5-1B OpenVINO accuracy validation (Phase 4).

MiniCPM5-1B is a decoder-only causal LM, so the meaningful accuracy signal is
the agreement of the next-token *logits* between the PyTorch reference and the
OpenVINO IR for the same prompt. This script reports, per the enablement
"cosine-similarity passable?" question:

  * mean / min per-token cosine similarity of the logit vectors
  * max absolute and max relative error
  * top-1 next-token agreement rate (argmax match)

Usage:
    python accuracy_check.py --model ./ov_minicpm \
        --reference openbmb/MiniCPM-1B-sft-bf16 \
        [--device CPU] [--precision f32] [--prompt "Hello"]

CI exit codes:
  0  All metrics within tolerance (result is "passable")
  1  One or more metrics exceed tolerance (needs improvement)
  2  Model load or inference error
"""
import argparse
import sys

import numpy as np

# Per-element error tolerance (logits). f16 path is looser because the IR math
# runs in half precision.
TOLERANCE = {
    "f32": {"atol": 1e-4, "rtol": 1e-3},
    "f16": {"atol": 1e-2, "rtol": 1e-2},
}

# Distribution-level gates. For an LLM these matter far more than raw atol/rtol
# on logits, which can be large in magnitude. A model is "passable" when the
# logit directions match closely and the predicted tokens agree.
COSINE_MIN = 0.999          # minimum acceptable per-token cosine similarity
TOP1_AGREEMENT_MIN = 0.99   # fraction of positions whose argmax token matches

DEFAULT_PROMPT = "OpenVINO is an open-source toolkit for optimizing and deploying AI inference."


def _cosine_per_token(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity over the vocab axis. Shapes: [seq, vocab]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.sum(a * b, axis=-1)
    den = (np.linalg.norm(a, axis=-1) + 1e-12) * (np.linalg.norm(b, axis=-1) + 1e-12)
    return num / den


def run_accuracy_check(model_path: str, reference: str, device: str, precision: str,
                       prompt: str, trust_remote_code: bool) -> int:
    tol = TOLERANCE[precision]

    try:
        import torch
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        print(f"[ERROR] Missing dependency: {exc}\n"
              "Install: pip install 'optimum-intel[openvino]' torch transformers",
              file=sys.stderr)
        return 2

    # --- Load OpenVINO model ---
    try:
        ov_config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "f32" else {}
        ov_model = OVModelForCausalLM.from_pretrained(
            model_path, device=device, ov_config=ov_config,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        print(f"[ERROR] OV model load failed: {exc}", file=sys.stderr)
        return 2

    # --- Load PyTorch reference ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(reference, trust_remote_code=trust_remote_code)
        pt_model = AutoModelForCausalLM.from_pretrained(
            reference, torch_dtype=torch.float32, trust_remote_code=trust_remote_code,
        ).eval()
    except Exception as exc:
        print(f"[ERROR] PyTorch reference load failed: {exc}", file=sys.stderr)
        return 2

    inputs = tokenizer(prompt, return_tensors="pt")

    # --- Reference logits ---
    with torch.no_grad():
        pt_logits = pt_model(**inputs).logits[0].float().numpy()  # [seq, vocab]

    # --- OV logits ---
    ov_out = ov_model(**inputs)
    ov_logits = np.asarray(ov_out.logits)[0].astype(np.float32)   # [seq, vocab]

    if pt_logits.shape != ov_logits.shape:
        print(f"[ERROR] Logit shape mismatch: pt={pt_logits.shape} ov={ov_logits.shape}",
              file=sys.stderr)
        return 1

    # --- Metrics ---
    abs_err = np.abs(pt_logits - ov_logits)
    rel_err = abs_err / (np.abs(pt_logits) + 1e-8)
    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())

    cos = _cosine_per_token(pt_logits, ov_logits)
    cos_mean = float(cos.mean())
    cos_min = float(cos.min())

    top1_match = float((pt_logits.argmax(-1) == ov_logits.argmax(-1)).mean())

    within_atol = max_abs <= tol["atol"] + tol["rtol"] * float(np.abs(pt_logits).max())
    cosine_ok = cos_min >= COSINE_MIN
    top1_ok = top1_match >= TOP1_AGREEMENT_MIN
    passed = cosine_ok and top1_ok

    print("=== MiniCPM5-1B accuracy report ===")
    print(f"prompt tokens         : {pt_logits.shape[0]}")
    print(f"precision path        : {precision}  (atol={tol['atol']}, rtol={tol['rtol']})")
    print(f"max abs error (logits): {max_abs:.6g}  [within_atol={within_atol}]")
    print(f"max rel error (logits): {max_rel:.6g}")
    print(f"cosine sim  mean / min: {cos_mean:.6f} / {cos_min:.6f}  "
          f"[>= {COSINE_MIN} -> {cosine_ok}]")
    print(f"top-1 token agreement : {top1_match:.4f}  "
          f"[>= {TOP1_AGREEMENT_MIN} -> {top1_ok}]")
    print(f"RESULT                : {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to OpenVINO IR directory")
    parser.add_argument("--reference", default="openbmb/MiniCPM-1B-sft-bf16",
                        help="HuggingFace id / path of the PyTorch reference")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--precision", default="f32", choices=["f32", "f16"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()
    try:
        sys.exit(run_accuracy_check(args.model, args.reference, args.device,
                                    args.precision, args.prompt, args.trust_remote_code))
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"[ERROR] Unhandled failure: {exc}", file=sys.stderr)
        sys.exit(2)
