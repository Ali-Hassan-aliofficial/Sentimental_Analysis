import streamlit as st
from transformers import pipeline
import shap

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

st.title("AI Sentiment Explainer (SHAP Powered)")

text = st.text_area(
    "Enter text",
    value="I love machine learning but sometimes it is frustrating"
)

if st.button("Analyze"):

    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    st.subheader("Final Prediction")
    st.success(f"{label} ({score:.2f})")

    # -----------------------------
    # SHAP explanation
    # -----------------------------
    st.subheader("Why this prediction? (AI explanation)")

    explainer = shap.Explainer(classifier)
    shap_values = explainer([text])

    tokens = shap_values.data[0]
    values = shap_values.values[0]

    # Positive / negative contributions
    positive_words = []
    negative_words = []

    for token, val in zip(tokens, values):

        if token in ["[CLS]", "[SEP]"]:
            continue

        if val > 0:
            positive_words.append((token, round(val, 3)))
        elif val < 0:
            negative_words.append((token, round(val, 3)))

    st.markdown("### Positive contributors")
    for w, v in sorted(positive_words, key=lambda x: -x[1]):
        st.write(f"🟢 {w} → +{v}")

    st.markdown("### Negative contributors")
    for w, v in sorted(negative_words, key=lambda x: x[1]):
        st.write(f"🔴 {w} → {v}")

    st.subheader("Raw output")
    st.write(result)
