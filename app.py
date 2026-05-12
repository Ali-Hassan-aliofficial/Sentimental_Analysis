import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    vader = SentimentIntensityAnalyzer()

    return sentiment_model, vader


sentiment_model, vader = load_models()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Hybrid Sentiment AI (DistilBERT + VADER Explainability)")

text = st.text_area(
    "Enter text",
    value="I love the product but the experience was frustrating and confusing."
)

# -----------------------------
# Run
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    # -------------------------
    # 1. DistilBERT (FINAL SENTIMENT)
    # -------------------------
    result = sentiment_model(text)[0]

    label = result["label"]
    score = result["score"]

    st.subheader("📊 Final Sentiment (DistilBERT)")
    st.success(f"{label} ({score:.2f})")

    # -------------------------
    # 2. VADER (WORD INSIGHT)
    # -------------------------
    st.subheader("🔍 Word-Level Insight (VADER)")

    words = text.lower().split()

    pos_words = []
    neg_words = []

    for word in words:
        score = vader.lexicon.get(word)

        if score is not None:
            if score > 0:
                pos_words.append(word)
            elif score < 0:
                neg_words.append(word)

    st.write("Positive words:", pos_words if pos_words else "None")
    st.write("Negative words:", neg_words if neg_words else "None")

    # -------------------------
    # 3. Explanation
    # -------------------------
    st.subheader("💡 Why this result?")

    explanation = f"""
The model predicted **{label}** because:

- DistilBERT analyzed the full sentence context.
- It detected overall sentiment strength ({score:.2f} confidence).
- Word-level signals show:
    - Positive cues: {pos_words if pos_words else "none"}
    - Negative cues: {neg_words if neg_words else "none"}

Final decision is based on contextual understanding, not just keywords.
"""

    st.write(explanation)
