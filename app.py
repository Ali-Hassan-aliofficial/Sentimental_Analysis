import streamlit as st
from transformers import pipeline

# -----------------------------
# Load robust sentiment model
# -----------------------------
@st.cache_resource
def load_sentiment():
    return pipeline(
        "text-classification",
        model="tabularisai/robust-sentiment-analysis",
        top_k=None
    )

sentiment_model = load_sentiment()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Robust Sentiment Analysis App")

text = st.text_area(
    "Enter text",
    value="I love the product but the experience was frustrating and confusing."
)

# -----------------------------
# Run analysis
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    result = sentiment_model(text)

    st.subheader("📊 Results")

    # result format: list of labels with scores
    for item in result[0]:
        st.write(f"{item['label']} → {item['score']:.3f}")

    best = max(result[0], key=lambda x: x["score"])

    st.success(f"Final Sentiment: {best['label']} ({best['score']:.2f})")
