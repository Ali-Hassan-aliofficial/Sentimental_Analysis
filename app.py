import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models():

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    explainer = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=80
    )

    vader = SentimentIntensityAnalyzer()

    return sentiment_model, explainer, vader


sentiment_model, explainer, vader = load_models()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Explainable Sentiment AI")

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
    # 1. Sentiment prediction
    # -----------------------------
    result = sentiment_model(text)[0]

    label = result["label"]
    confidence = result["score"]

    # -----------------------------
    # 2. Generate explanation (FLAN-T5)
    # -----------------------------
    prompt = f"""
Explain in simple terms why this text is {label.lower()} sentiment.

Text: {text}

Give a short human-like explanation.
"""

    explanation = explainer(prompt)[0]["generated_text"]

    # -----------------------------
    # 3. VADER word signals (optional insight)
    # -----------------------------
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

    # -----------------------------
    # OUTPUT (clean product style)
    # -----------------------------
    st.subheader("📊 Result")

    st.success(f"Sentiment: {label}")
    st.info(f"Confidence: {confidence}")

    st.subheader("💡 Explanation")

    st.write(explanation)

    # -----------------------------
    # Optional insights (hidden intelligence layer)
    # -----------------------------
    if pos_words or neg_words:
        st.subheader("🔍 Key Signals")

        if pos_words:
            st.write("Positive cues:", pos_words)

        if neg_words:
            st.write("Negative cues:", neg_words)
