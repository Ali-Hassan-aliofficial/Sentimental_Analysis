import streamlit as st
from transformers import pipeline
import shap
import numpy as np

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

st.title("🧠 Sentiment Analysis with AI Explanation (SHAP)")

text = st.text_area(
    "Enter your text",
    value="I love machine learning but sometimes it is very frustrating and confusing"
)

# -----------------------------
# Run Analysis
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    # -----------------------------
    # 1. Sentiment Prediction
    # -----------------------------
    with st.spinner("Running sentiment model..."):
        result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    st.subheader("📊 Final Prediction")
    st.success(f"{label} ({score:.2f})")

    # -----------------------------
    # 2. SHAP Explanation (IMPORTANT PART)
    # -----------------------------
    st.subheader("🧠 Why did the model predict this?")

    with st.spinner("Generating explanation (SHAP)... please wait ⏳"):

        explainer = shap.Explainer(classifier)
        shap_values = explainer([text])

        tokens = shap_values.data[0]
        values = shap_values.values

        # -----------------------------
        # FIX: handle SHAP output shape safely
        # -----------------------------
        values = np.array(values)

        # Case: multi-class output (we take positive class)
        if len(values.shape) == 3:
            values = values[0][:, 1]
        else:
            values = values[0]

        positive_words = []
        negative_words = []

        for token, val in zip(tokens, values):

            # skip special tokens
            if token in ["[CLS]", "[SEP]"]:
                continue

            val = float(val)

            if val > 0:
                positive_words.append((token, val))
            elif val < 0:
                negative_words.append((token, val))

    # -----------------------------
    # 3. Display explanations
    # -----------------------------
    st.markdown("### 🟢 Positive contributors")

    if positive_words:
        for w, v in sorted(positive_words, key=lambda x: -x[1]):
            st.write(f"🟢 {w} → +{v:.3f}")
    else:
        st.write("No strong positive contributors found.")

    st.markdown("### 🔴 Negative contributors")

    if negative_words:
        for w, v in sorted(negative_words, key=lambda x: x[1]):
            st.write(f"🔴 {w} → {v:.3f}")
    else:
        st.write("No strong negative contributors found.")

    # -----------------------------
    # 4. Human-like explanation
    # -----------------------------
    st.subheader("💡 AI Explanation")

    if label == "POSITIVE":

        if positive_words and negative_words:
            st.info("Mixed emotions detected, but positive signals dominate the model decision.")

        elif positive_words:
            st.info("The model focused mainly on positive words influencing the prediction.")

        else:
            st.info("The model interpreted overall tone as positive.")

    else:

        if positive_words and negative_words:
            st.warning("Mixed emotions detected, but negative signals dominate the model decision.")

        elif negative_words:
            st.warning("Negative words strongly influenced the prediction.")

        else:
            st.warning("The model interpreted overall tone as negative.")

    # -----------------------------
    # 5. Raw output
    # -----------------------------
    st.subheader("🔍 Raw Model Output")
    st.write(result)
