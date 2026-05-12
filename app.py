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
# Run analysis
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    # -----------------------------
    # 1. DistilBERT SENTIMENT
    # -----------------------------
    result = sentiment_model(text)[0]

    label = result["label"]
    confidence = result["score"]

    st.subheader("📊 Final Sentiment (AI Model)")
    st.success(f"{label} ({confidence:.2f})")

    # -----------------------------
    # 2. VADER WORD ANALYSIS
    # -----------------------------
    words = text.lower().split()

    pos_words = []
    neg_words = []

    for word in words:
        vader_score = vader.lexicon.get(word)

        if vader_score is not None:
            if vader_score > 0:
                pos_words.append(word)
            elif vader_score < 0:
                neg_words.append(word)

    st.subheader("🔍 Word-Level Explanation (VADER)")

    st.write("Positive words:", pos_words if pos_words else "None")
    st.write("Negative words:", neg_words if neg_words else "None")

    # -----------------------------
    # 3. Explanation
    # -----------------------------
    st.subheader("💡 Why this prediction?")

    explanation = f"""
The final sentiment is **{label}** with confidence **{confidence:.2f}**.

Reasoning:
- DistilBERT analyzed full context of the sentence.
- It does NOT rely only on keywords.
- VADER detected keyword signals:
    - Positive: {pos_words if pos_words else "none"}
    - Negative: {neg_words if neg_words else "none"}

Final decision is based on contextual understanding + lexical signals.
"""

    st.write(explanation)
