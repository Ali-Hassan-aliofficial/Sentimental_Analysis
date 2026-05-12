import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# Load analyzer
# -----------------------------
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

analyzer = load_vader()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Sentiment + Word Explainer (Lexicon-Based)")

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
    # Sentiment score
    # -------------------------
    scores = analyzer.polarity_scores(text)

    st.subheader("📊 Sentiment Scores")
    st.write(scores)

    # -------------------------
    # Extract words
    # -------------------------
    words = text.lower().split()

    pos_words = []
    neg_words = []

    for word in words:
        score = analyzer.lexicon.get(word)

        if score is not None:
            if score > 0:
                pos_words.append(word)
            elif score < 0:
                neg_words.append(word)

    # -------------------------
    # Display results
    # -------------------------
    st.subheader("🔍 Positive Words Found")
    st.write(pos_words if pos_words else "None")

    st.subheader("🔍 Negative Words Found")
    st.write(neg_words if neg_words else "None")

    # -------------------------
    # Final decision
    # -------------------------
    compound = scores["compound"]

    if compound >= 0.05:
        final = "POSITIVE"
    elif compound <= -0.05:
        final = "NEGATIVE"
    else:
        final = "NEUTRAL"

    st.subheader("🧠 Final Sentiment")
    st.success(final)
