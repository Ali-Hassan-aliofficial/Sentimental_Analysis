import streamlit as st
from transformers import pipeline

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_explainer():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

sentiment_model = load_sentiment_model()
explainer_model = load_explainer()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 AI Sentiment Analyzer Pro")

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

    # -------------------------
    # 1. SENTIMENT PREDICTION
    # -------------------------
    result = sentiment_model(text)[0]

    label = result["label"]
    score = result["score"]

    st.subheader("📊 Sentiment Result")
    st.success(f"{label} ({score:.2f})")

    # -------------------------
    # 2. EXPLANATION MODEL
    # -------------------------
    prompt = f"""
Explain the sentiment in simple terms.

Text:
{text}

Sentiment:
{label}

Give:
- Reason
- Positive words
- Negative words
"""

    with st.spinner("Generating explanation..."):

        explanation = explainer_model(
            prompt,
            max_new_tokens=150,
            do_sample=False
        )[0]["generated_text"]

    st.subheader("💡 Explanation")
    st.write(explanation)
