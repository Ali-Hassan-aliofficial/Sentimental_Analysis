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
st.title("🧠 Sentiment Analysis AI")

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

    # -----------------------------
    # Sentiment prediction
    # -----------------------------
    result = sentiment_model(text)[0]

    label = result["label"]
    confidence = result["score"]  # full float, no rounding

    # -----------------------------
    # VADER word extraction (internal use only)
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
    # CLEAN OUTPUT (NO MODEL INFO)
    # -----------------------------
    st.subheader("📊 Result")

    st.success(f"Sentiment: {label}")
    st.info(f"Confidence: {confidence}")

    # -----------------------------
    # USER-FACING EXPLANATION (NO TECH DETAILS)
    # -----------------------------
    st.subheader("💡 Insight")

    if label == "NEGATIVE":
        msg = "The text contains stronger negative emotional tone than positive tone."
    elif label == "POSITIVE":
        msg = "The text contains stronger positive emotional tone than negative tone."
    else:
        msg = "The text shows a balanced or neutral emotional tone."

    st.write(msg)

    # -----------------------------
    # OPTIONAL: subtle word insight (NO labels like VADER)
    # -----------------------------
    if pos_words or neg_words:
        st.write("Key emotional signals were detected in the text.")

        if pos_words:
            st.write("Positive cues:", pos_words)

        if neg_words:
            st.write("Negative cues:", neg_words)
