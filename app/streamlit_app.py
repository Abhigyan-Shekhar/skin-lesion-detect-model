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

from question_engine import get_adaptive_questions, run_engine
from summary_generator import generate_summary, summary_to_text
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


def init_state() -> None:
    defaults = {
        "accepted_disclaimer": False,
        "patient_intake": {},
        "predictions_payload": None,
        "answers": {},
        "engine_output": None,
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


def run_checkpoint_inference(
    checkpoint_path: Path,
    image_path: Path,
    top_k: int,
) -> dict[str, Any]:
    try:
        import torch

        from dataset import build_transforms
        from models import build_model
        from utils import resolve_device
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Model inference dependencies are not installed. Run: pip install -r requirements.txt"
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    device = resolve_device(config.get("device", "auto"))

    model = build_model(
        model_name=config["model_name"],
        num_classes=len(class_to_idx),
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = build_transforms(image_size=int(config["image_size"]), train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=min(top_k, probs.shape[0]))

    predictions = [
        {"label": idx_to_class[int(index.item())], "probability": round(float(value.item()), 6)}
        for value, index in zip(values, indices)
    ]
    return {
        "top_predictions": predictions,
        "confidence_level": confidence_level(float(values[0].item())),
        "disclaimer": DISCLAIMER_TEXT,
    }


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
        "disclaimer": DISCLAIMER_TEXT,
    }


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
        chief_complaint = st.text_area(
            "Chief complaint in patient's words",
            value=st.session_state.patient_intake.get("chief_complaint", ""),
            height=90,
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
            "chief_complaint": chief_complaint,
        }
        st.success("Intake saved.")


def render_prediction_panel() -> None:
    st.subheader("Image And Model Prediction")
    uploaded = st.file_uploader("Upload lesion image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        checkpoint_text = st.text_input(
            "Checkpoint path",
            value=str(PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"),
        )
    with col_b:
        top_k = st.number_input("Top-k", min_value=1, max_value=10, value=5)

    if st.button("Run Checkpoint Inference", disabled=uploaded is None):
        checkpoint_path = Path(checkpoint_text)
        if not checkpoint_path.exists():
            st.warning("Checkpoint not found. Use manual predictions below until a model is trained.")
        else:
            suffix = Path(uploaded.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(uploaded.getbuffer())
                temp_path = Path(handle.name)
            try:
                st.session_state.predictions_payload = run_checkpoint_inference(
                    checkpoint_path=checkpoint_path,
                    image_path=temp_path,
                    top_k=int(top_k),
                )
                st.success("Model prediction generated.")
            except RuntimeError as exc:
                st.error(str(exc))
            finally:
                temp_path.unlink(missing_ok=True)

    st.divider()
    st.caption("Manual top-k entry is available for demo flow testing before a trained checkpoint exists.")
    manual_rows = []
    for index in range(3):
        col_label, col_prob = st.columns([3, 1])
        with col_label:
            label = st.text_input(
                f"Prediction {index + 1}",
                value=DEFAULT_PREDICTION_LABELS[min(index, len(DEFAULT_PREDICTION_LABELS) - 1)],
                key=f"manual_label_{index}",
            )
        with col_prob:
            probability = st.number_input(
                f"Probability {index + 1}",
                min_value=0.0,
                max_value=1.0,
                value=[0.42, 0.21, 0.13][index],
                step=0.01,
                key=f"manual_prob_{index}",
            )
        manual_rows.append({"label": label, "probability": probability})

    if st.button("Use Manual Predictions"):
        st.session_state.predictions_payload = normalize_manual_predictions(manual_rows)
        st.success("Manual predictions saved.")

    if st.session_state.predictions_payload:
        st.json(st.session_state.predictions_payload)


def render_questions() -> None:
    st.subheader("Adaptive OPD Questions")
    predictions_payload = st.session_state.predictions_payload
    if not predictions_payload:
        st.info("Add model or manual predictions first.")
        return

    questions = get_adaptive_questions(predictions_payload["top_predictions"])
    answers: dict[str, Any] = {}
    with st.form("adaptive_questions_form"):
        for question in questions:
            question_id = question["id"]
            default = st.session_state.answers.get(question_id)
            if question.get("type") == "adaptive":
                selected = st.radio(
                    question["text"],
                    ["Not answered", "Yes", "No"],
                    horizontal=True,
                    key=f"answer_{question_id}",
                    index={"Not answered": 0, True: 1, False: 2}.get(default, 0),
                )
                if selected == "Yes":
                    answers[question_id] = True
                elif selected == "No":
                    answers[question_id] = False
            else:
                value = st.text_input(
                    question["text"],
                    value="" if default in [None, True, False] else str(default),
                    key=f"answer_{question_id}",
                )
                if value.strip():
                    answers[question_id] = value.strip()
        submitted = st.form_submit_button("Score Answers")

    if submitted:
        st.session_state.answers = answers
        st.session_state.engine_output = run_engine(predictions_payload, answers=answers)
        st.success("Adaptive scoring complete.")

    if st.session_state.engine_output:
        scoring = st.session_state.engine_output["scoring"]
        st.metric("Urgency", scoring["urgency_level"])
        st.write(scoring["doctor_review_priority"])
        st.json(scoring)


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
            image_quality_warning="Not automatically assessed in demo app.",
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
