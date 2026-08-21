from __future__ import annotations

import argparse
import json

from coarse_fusion import predict_coarse_label
from fusion import build_combined_payload
from question_engine import run_engine
from router import run_dual_model_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DermaCon-IN and HAM10000 checkpoints on one image and build a fused question payload."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--opd_checkpoint", required=True, help="Path to DermaCon-IN broad OPD checkpoint")
    parser.add_argument("--lesion_checkpoint", required=True, help="Path to HAM10000 lesion checkpoint")
    parser.add_argument("--top_k_opd", type=int, default=5)
    parser.add_argument("--top_k_lesion", type=int, default=5)
    parser.add_argument("--answers", default=None, help="Optional JSON string/file of adaptive answers")
    parser.add_argument("--output", default=None, help="Optional path to save the fused JSON output")
    parser.add_argument(
        "--coarse_fusion",
        action="store_true",
        help="Fuse 3-class image probs with history answers (requires coarse-class checkpoint)",
    )
    parser.add_argument("--fusion_alpha", type=float, default=0.6, help="Image weight for coarse fusion")
    return parser.parse_args()


def load_json_arg(value: str) -> dict:
    import json
    from pathlib import Path

    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def main() -> None:
    args = parse_args()
    dual_model_result = run_dual_model_inference(
        image_path=args.image,
        opd_checkpoint=args.opd_checkpoint,
        lesion_checkpoint=args.lesion_checkpoint,
        top_k_opd=args.top_k_opd,
        top_k_lesion=args.top_k_lesion,
    )
    combined_payload = build_combined_payload(dual_model_result)
    answers = load_json_arg(args.answers) if args.answers else {}
    engine_output = run_engine(combined_payload, answers=answers)

    result = {
        "dual_model_result": dual_model_result,
        "combined_payload": combined_payload,
        "engine_output": engine_output,
    }

    if args.coarse_fusion:
        opd_branch = dual_model_result.get("branches", {}).get("opd", {})
        top_preds = opd_branch.get("top_predictions", [])
        image_probs = {row["label"]: float(row["probability"]) for row in top_preds}
        fused_label, fused_probs = predict_coarse_label(image_probs, answers, alpha=args.fusion_alpha)
        result["coarse_fusion"] = {
            "image_only_probs": image_probs,
            "fused_label": fused_label,
            "fused_probs": fused_probs,
            "alpha": args.fusion_alpha,
        }

    if args.output:
        from utils import write_json

        write_json(args.output, result)

    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
