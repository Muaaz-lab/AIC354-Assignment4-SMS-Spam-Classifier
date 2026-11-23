"""Streamlit interface for the saved SMS spam classifier."""

from pathlib import Path

import joblib
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_sms_spam_model.joblib"

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("SMS Spam Classifier")
st.write(
    "Enter a message to classify it as **ham** (legitimate) or **spam**. "
    "This classroom demo uses a TF-IDF text model trained on the UCI SMS Spam Collection."
)

st.warning(
    "Coursework demonstration only. The dataset is limited and dated; do not use this "
    "classifier to filter important messages or make security decisions."
)

message = st.text_area(
    "Message text",
    placeholder="Example: Congratulations! You have won a prize. Reply NOW to claim...",
    height=150,
    max_chars=5_000,
)

if st.button("Classify message", type="primary", use_container_width=True):
    if not message.strip():
        st.info("Enter a message first.")
    elif not MODEL_PATH.exists():
        st.error("The model file is missing. Run `python train_models.py` from this folder.")
    else:
        model = load_model()
        prediction = str(model.predict([message])[0])
        if prediction == "spam":
            st.error("Prediction: **SPAM**")
        else:
            st.success("Prediction: **HAM (legitimate)**")
        st.caption(
            "The prediction is a model estimate. Review unfamiliar messages carefully and "
            "never share passwords or verification codes in response to a message."
        )

st.divider()
st.caption("AIC354 · Assignment 4 · Text classification with TF-IDF and scikit-learn")

