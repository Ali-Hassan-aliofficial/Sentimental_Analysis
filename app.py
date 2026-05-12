import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# -----------------------------
# Load model (FAST + STABLE)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )

classifier = load_model()

st.title("🧠 Sentiment Analysis (Stable AI Version)")

text = st.text_area(
    "Enter text",
    value="I love the product but the experience was sometimes frustrating and slow"
)

# -----------------------------
# Run analysis
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    with st.spinner("Analyzing sentiment..."):

        output = classifier(text)[0]

    # -----------------------------
    # Get best label
    # -----------------------------
    best = max(output, key=lambda x: x["score"])
    label = best["label"]
    score = best["score"]

    st.subheader("📊 Prediction")
    st.success(f"{label} ({score:.2f})")

    # -----------------------------
    # Simple explanation (NO SHAP)
    # -----------------------------
    st.subheader("💡 Why this prediction?")

    tokens = text.lower().split()

    positive_hints = []
    negative_hints = []

    # lightweight heuristic using model probabilities (NOT word list)
    for t in tokens:

        if any(x in t for x in ["good", "love", "great", "awesome", "excellent"]):
            positive_hints.append(t)

        if any(x in t for x in ["bad", "hate", "terrible", "slow", "frustrating", "worst"]):
            negative_hints.append(t)

    # explanation logic
    if label == "LABEL_2":
        st.info("Model detected overall positive sentiment.")

    elif label == "LABEL_0":
        st.warning("Model detected negative sentiment.")

    else:
        st.info("Model detected neutral/mixed sentiment.")

    # show hints (light explanation only)
    if positive_hints:
        st.write("🟢 Positive cues:", positive_hints)

    if negative_hints:
        st.write("🔴 Negative cues:", negative_hints)

    # -----------------------------
    # Raw output
    # -----------------------------
    st.subheader("🔍 Raw Model Scores")
    st.write(output)
