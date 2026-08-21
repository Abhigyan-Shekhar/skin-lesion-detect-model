from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fusion import build_combined_payload
from coarse_fusion import predict_coarse_label
from image_quality import assess_image_quality, image_quality_warning_text
from router import run_modality_gated_inference
from summary_generator import generate_summary, summary_to_text
from triage_state import build_belief_state
from utils import DISCLAIMER_TEXT


DEFAULT_PREDICTION_LABELS = [
    "mel",
    "nv",
    "bcc",
    "bkl",
    "akiec",
    "df",
    "vasc",
    "tinea corporis",
    "eczema",
    "impetigo",
]

DEFAULT_OPD_LABELS = [
    "Pigmentary Disorders",
    "Inflammatory Disorders",
    "Infectious Disorders",
    "Other skin disorders",
]

DEFAULT_LESION_LABELS = [
    "mel",
    "nv",
    "bcc",
    "bkl",
    "akiec",
    "df",
    "vasc",
]


def init_state() -> None:
    defaults = {
        "accepted_disclaimer": False,
        "patient_intake": {},
        "predictions_payload": None,
        "answers": {},
        "engine_output": None,
        "belief_state": None,
        "image_quality": None,
        "modality": "clinical",
        "summary": None,
        "summary_text": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def confidence_level(max_prob: float) -> str:
    if max_prob >= 0.75:
        return "high"
    if max_prob >= 0.45:
        return "moderate"
    return "low"


def normalize_manual_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [
        {
            "label": row["label"].strip(),
            "probability": round(float(row["probability"]), 6),
        }
        for row in rows
        if row["label"].strip() and float(row["probability"]) > 0
    ]
    predictions.sort(key=lambda item: item["probability"], reverse=True)
    max_prob = predictions[0]["probability"] if predictions else 0.0
    return {
        "top_predictions": predictions,
        "confidence_level": confidence_level(max_prob),
        "max_probability": round(max_prob, 6),
        "disclaimer": DISCLAIMER_TEXT,
    }


def normalize_manual_branch_predictions(rows: list[dict[str, Any]], branch_name: str, modality: str) -> dict[str, Any]:
    return build_combined_payload(
        {
            "mode": "single_model",
            "modality": modality,
            "image_path": "manual-demo",
            "branches": {branch_name: normalize_manual_predictions(rows)},
        }
    )


def reset_reasoning_state() -> None:
    st.session_state.answers = {}
    st.session_state.engine_output = None
    st.session_state.belief_state = None
    st.session_state.summary = None
    st.session_state.summary_text = ""


def render_disclaimer() -> None:
    st.title("Dermatology OPD Triage Research Demo")
    st.error(
        "This is a research prototype and is not for medical diagnosis or treatment. "
        "It is not a medical device, not for clinical deployment, and requires doctor review."
    )
    st.markdown(
        """
        - Research use only
        - Non-commercial use only
        - Not for autonomous diagnosis
        - No treatment recommendations
        - All outputs require review by a qualified clinician
        """
    )
    st.checkbox(
        "I understand this demo is research-only and not for medical use.",
        key="accepted_disclaimer",
    )


def render_patient_intake() -> None:
    st.subheader("Patient Intake")
    with st.form("patient_intake_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            patient_id = st.text_input("Anonymized patient ID", value=st.session_state.patient_intake.get("patient_id", ""))
            age = st.number_input("Age", min_value=0, max_value=120, value=int(st.session_state.patient_intake.get("age", 30)))
        with col_b:
            sex = st.selectbox(
                "Sex",
                ["Not specified", "Female", "Male", "Other"],
                index=0,
            )
            region = st.text_input("Region", value=st.session_state.patient_intake.get("region", ""))
        with col_c:
            occupation = st.text_input("Occupation", value=st.session_state.patient_intake.get("occupation", ""))
            education = st.text_input("Education", value=st.session_state.patient_intake.get("education", ""))
        body_part_affected = st.text_input(
            "Location / body part affected",
            value=st.session_state.patient_intake.get("body_part_affected", ""),
        )
        chief_complaint = st.text_area(
            "Chief complaint in patient's words",
            value=st.session_state.patient_intake.get("chief_complaint", ""),
            height=90,
        )
        st.markdown("**Lesion metadata from image review / model**")
        meta_a, meta_b, meta_c, meta_d = st.columns(4)
        with meta_a:
            lesion_color = st.text_input("Color", value=st.session_state.patient_intake.get("lesion_color", ""))
            lesion_shape = st.text_input("Shape", value=st.session_state.patient_intake.get("lesion_shape", ""))
        with meta_b:
            lesion_border = st.text_input("Border", value=st.session_state.patient_intake.get("lesion_border", ""))
            lesion_size = st.text_input("Size", value=st.session_state.patient_intake.get("lesion_size", ""))
        with meta_c:
            lesion_pattern = st.text_input("Pattern", value=st.session_state.patient_intake.get("lesion_pattern", ""))
            lesion_pigmentation = st.text_input("Pigmentation", value=st.session_state.patient_intake.get("lesion_pigmentation", ""))
        with meta_d:
            lesion_surface_change = st.text_input(
                "Scaling / crusting / ulceration",
                value=st.session_state.patient_intake.get("lesion_surface_change", ""),
            )
            lesion_count = st.selectbox(
                "Single or multiple lesions",
                ["Not specified", "Single", "Multiple"],
                index=["Not specified", "Single", "Multiple"].index(
                    st.session_state.patient_intake.get("lesion_count", "Not specified")
                    if st.session_state.patient_intake.get("lesion_count", "Not specified") in {"Not specified", "Single", "Multiple"}
                    else "Not specified"
                ),
            )
        submitted = st.form_submit_button("Save Intake")

    if submitted:
        st.session_state.patient_intake = {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "region": region,
            "occupation": occupation,
            "education": education,
            "body_part_affected": body_part_affected,
            "chief_complaint": chief_complaint,
            "lesion_color": lesion_color,
            "lesion_shape": lesion_shape,
            "lesion_border": lesion_border,
            "lesion_size": lesion_size,
            "lesion_pattern": lesion_pattern,
            "lesion_pigmentation": lesion_pigmentation,
            "lesion_surface_change": lesion_surface_change,
            "lesion_count": lesion_count,
        }
        st.success("Intake saved.")


def render_prediction_panel() -> None:
    st.subheader("Image And Model Prediction")
    modality = st.radio(
        "Image modality",
        ["Clinical photograph", "Dermoscopic image", "Both images", "Unknown modality"],
        horizontal=True,
    )
    modality_key = {
        "Clinical photograph": "clinical",
        "Dermoscopic image": "dermoscopic",
        "Both images": "both",
        "Unknown modality": "unknown",
    }[modality]
    st.session_state.modality = modality_key

    clinical_upload = None
    dermoscopic_upload = None
    if modality_key in {"clinical", "both", "unknown"}:
        clinical_upload = st.file_uploader("Upload clinical photograph", type=["jpg", "jpeg", "png"], key="clinical_upload")
        if clinical_upload:
            image = Image.open(clinical_upload).convert("RGB")
            st.image(image, caption="Clinical photograph", use_container_width=True)
    if modality_key in {"dermoscopic", "both"}:
        dermoscopic_upload = st.file_uploader("Upload dermoscopic image", type=["jpg", "jpeg", "png"], key="dermoscopic_upload")
        if dermoscopic_upload:
            image = Image.open(dermoscopic_upload).convert("RGB")
            st.image(image, caption="Dermoscopic image", use_container_width=True)

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        opd_checkpoint_text = st.text_input(
            "DermaCon-IN checkpoint",
            value=str(PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"),
        )
    with col_b:
        lesion_checkpoint_text = st.text_input(
            "HAM10000 checkpoint",
            value=str(PROJECT_ROOT / "outputs_ham10000" / "checkpoints" / "best.pt"),
        )
    with col_c:
        top_k = st.number_input("Top-k", min_value=1, max_value=10, value=5)

    upload_ready = (
        (modality_key == "clinical" and clinical_upload is not None)
        or (modality_key == "dermoscopic" and dermoscopic_upload is not None)
        or (modality_key == "both" and clinical_upload is not None and dermoscopic_upload is not None)
        or modality_key == "unknown"
    )
    if st.button("Run Modality-Gated Inference", disabled=not upload_ready):
        opd_checkpoint = Path(opd_checkpoint_text)
        lesion_checkpoint = Path(lesion_checkpoint_text)
        needs_opd = modality_key in {"clinical", "both"}
        needs_lesion = modality_key in {"dermoscopic", "both"}
        if (needs_opd and not opd_checkpoint.exists()) or (needs_lesion and not lesion_checkpoint.exists()):
            st.warning("The modality-selected checkpoint was not found. Use the manual prediction demo below until the model is trained.")
        elif modality_key == "unknown":
            reset_reasoning_state()
            st.session_state.predictions_payload = None
            st.session_state.image_quality = {"accepted": False, "warnings": ["Unsupported or unknown image modality."], "measurements": {}}
            st.warning("Unknown modality: request the correct image type before model inference.")
        else:
            temp_paths: dict[str, Path] = {}
            try:
                if clinical_upload is not None:
                    suffix = Path(clinical_upload.name).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                        handle.write(clinical_upload.getbuffer())
                        temp_paths["clinical"] = Path(handle.name)
                if dermoscopic_upload is not None:
                    suffix = Path(dermoscopic_upload.name).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                        handle.write(dermoscopic_upload.getbuffer())
                        temp_paths["dermoscopic"] = Path(handle.name)

                quality_results = {
                    branch: assess_image_quality(path)
                    for branch, path in temp_paths.items()
                }
                st.session_state.image_quality = {
                    "accepted": all(item["accepted"] for item in quality_results.values()),
                    "warnings": [
                        f"{branch}: {warning}"
                        for branch, result in quality_results.items()
                        for warning in result.get("warnings", [])
                    ],
                    "measurements": {
                        branch: result.get("measurements", {})
                        for branch, result in quality_results.items()
                    },
                } if quality_results else None
                if st.session_state.image_quality and not st.session_state.image_quality["accepted"]:
                    reset_reasoning_state()
                    st.warning(image_quality_warning_text(st.session_state.image_quality))
                    return

                model_result = run_modality_gated_inference(
                    modality=modality_key,
                    clinical_image_path=temp_paths.get("clinical"),
                    dermoscopic_image_path=temp_paths.get("dermoscopic"),
                    opd_checkpoint=opd_checkpoint,
                    lesion_checkpoint=lesion_checkpoint,
                    top_k=int(top_k),
                )
                if model_result.get("mode") == "abstain":
                    reset_reasoning_state()
                    st.session_state.predictions_payload = None
                    st.warning(model_result["reason"])
                else:
                    st.session_state.predictions_payload = build_combined_payload(model_result)
                    reset_reasoning_state()
                    st.success("Modality-appropriate prediction generated.")
            except RuntimeError as exc:
                st.error(str(exc))
            finally:
                for temp_path in temp_paths.values():
                    temp_path.unlink(missing_ok=True)

    st.divider()
    st.caption("Manual entry is available for demo flow testing before checkpoints exist.")
    manual_branch = st.radio(
        "Manual prediction branch",
        ["Clinical OPD branch", "Dermoscopy lesion branch"],
        horizontal=True,
    )
    branch_name = "opd" if manual_branch == "Clinical OPD branch" else "ham10000"
    manual_modality = "clinical" if branch_name == "opd" else "dermoscopic"
    label_options = DEFAULT_OPD_LABELS if branch_name == "opd" else DEFAULT_LESION_LABELS
    default_probs = [0.62, 0.20, 0.12] if branch_name == "opd" else [0.41, 0.24, 0.19]
    st.markdown(f"**Manual {manual_branch}**")
    manual_rows = []
    for index in range(3):
        col_label, col_prob = st.columns([3, 1])
        with col_label:
            label = st.text_input(
                f"Prediction {index + 1}",
                value=label_options[min(index, len(label_options) - 1)],
                key=f"manual_{branch_name}_label_{index}",
            )
        with col_prob:
            probability = st.number_input(
                f"Probability {index + 1}",
                min_value=0.0,
                max_value=1.0,
                value=default_probs[index],
                step=0.01,
                key=f"manual_{branch_name}_prob_{index}",
            )
        manual_rows.append({"label": label, "probability": probability})

    if st.button("Use Manual Predictions"):
        st.session_state.predictions_payload = normalize_manual_branch_predictions(manual_rows, branch_name, manual_modality)
        st.session_state.modality = manual_modality
        st.session_state.image_quality = {"accepted": True, "warnings": [], "measurements": {}}
        reset_reasoning_state()
        st.success("Manual predictions saved.")

    if st.session_state.predictions_payload:
        st.json(st.session_state.predictions_payload)


def render_questions() -> None:
    st.subheader("Adaptive OPD Questions")
    predictions_payload = st.session_state.predictions_payload
    if not predictions_payload:
        st.info("Add model or manual predictions first.")
        return

    belief_state = build_belief_state(
        predictions_payload,
        answers=st.session_state.answers,
        patient_context=st.session_state.patient_intake,
        image_quality=st.session_state.image_quality,
        modality=st.session_state.modality,
    )
    st.session_state.belief_state = belief_state
    st.session_state.engine_output = belief_state["engine_output"]

    scoring = belief_state["engine_output"]["scoring"]
    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.metric("Questions asked", len(belief_state["questions_already_asked"]))
    with metric_b:
        st.metric("Uncertainty", belief_state["uncertainty"])
    with metric_c:
        st.metric("Urgency", scoring["urgency_level"])
    st.write(scoring["doctor_review_priority"])

    next_question = belief_state.get("next_question")
    if next_question:
        st.markdown("**Next question**")
        st.caption(f"Utility: {next_question['utility']} | Section: {next_question.get('section', 'history')}")
        with st.form(f"single_question_{next_question['id']}"):
            if next_question.get("type") == "general" and next_question["id"] not in {
                "hypertension",
                "previous_hospital_admission",
            }:
                value = st.text_input(next_question["text"])
                submitted = st.form_submit_button("Save Answer")
                if submitted and value.strip():
                    st.session_state.answers[next_question["id"]] = value.strip()
                    st.rerun()
            else:
                selected = st.radio(next_question["text"], ["Yes", "No"], horizontal=True)
                submitted = st.form_submit_button("Save Answer")
                if submitted:
                    st.session_state.answers[next_question["id"]] = selected == "Yes"
                    st.rerun()
    else:
        st.info(f"Questioning stopped: {', '.join(belief_state['stop_reasons']) or 'complete'}.")

    if st.button("Reset Adaptive Answers"):
        reset_reasoning_state()
        st.rerun()

    with st.expander("Belief state", expanded=False):
        st.json(
            {
                "positive_findings": belief_state["positive_findings"],
                "negative_findings": belief_state["negative_findings"],
                "red_flags": belief_state["red_flags"],
                "current_differential": belief_state["current_differential"][:5],
                "candidate_questions": belief_state["candidate_questions"],
                "stop_reasons": belief_state["stop_reasons"],
            }
        )

    if st.session_state.engine_output:
        with st.expander("Scoring payload", expanded=False):
            st.json(scoring)

        payload = st.session_state.predictions_payload
        top_preds = payload.get("top_predictions", payload) if isinstance(payload, dict) else payload
        if isinstance(top_preds, list) and top_preds:
            image_probs = {
                str(row["label"]): float(row.get("probability", row.get("prob", 0.0)))
                for row in top_preds
            }
            fused_label, fused_probs = predict_coarse_label(
                image_probs, st.session_state.answers, alpha=0.6
            )
            st.subheader("3-class fusion (image + history)")
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Image-only top label")
                st.write(max(image_probs, key=image_probs.get))
            with col_b:
                st.caption("Fused triage label")
                st.write(fused_label)
            st.json({"fused_probs": fused_probs})


def render_summary() -> None:
    st.subheader("Doctor Summary")
    if not st.session_state.patient_intake:
        st.info("Save patient intake first.")
        return
    if not st.session_state.predictions_payload:
        st.info("Add model or manual predictions first.")
        return
    if not st.session_state.engine_output:
        st.info("Score adaptive answers first.")
        return

    if st.button("Generate Doctor Summary"):
        st.session_state.summary = generate_summary(
            patient_intake=st.session_state.patient_intake,
            predictions_payload=st.session_state.predictions_payload,
            engine_output=st.session_state.engine_output,
            answers=st.session_state.answers,
            image_quality_warning=image_quality_warning_text(st.session_state.image_quality),
        )
        st.session_state.summary_text = summary_to_text(st.session_state.summary)

    if st.session_state.summary:
        st.text_area("Structured OPD case note", st.session_state.summary_text, height=420)
        st.download_button(
            "Export JSON",
            data=json.dumps(st.session_state.summary, indent=2),
            file_name="opd_summary.json",
            mime="application/json",
        )
        st.download_button(
            "Export TXT",
            data=st.session_state.summary_text,
            file_name="opd_summary.txt",
            mime="text/plain",
        )


def main() -> None:
    st.set_page_config(page_title="Derm OPD Triage Research Demo", layout="wide")
    init_state()
    render_disclaimer()

    if not st.session_state.accepted_disclaimer:
        st.stop()

    tabs = st.tabs(["Patient Intake", "Image Prediction", "Adaptive Questions", "Doctor Summary"])
    with tabs[0]:
        render_patient_intake()
    with tabs[1]:
        render_prediction_panel()
    with tabs[2]:
        render_questions()
    with tabs[3]:
        render_summary()


if __name__ == "__main__":
    main()
