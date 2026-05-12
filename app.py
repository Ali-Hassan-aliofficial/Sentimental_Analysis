import streamlit as st
from transformers import pipeline
import re

@st.cache_resource
def load_classifier():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_classifier()

st.title("Sentiment Analysis App (Improved)")

st.write("Paste a paragraph and get sentence-level + overall sentiment analysis.")

text = st.text_area(
    "Your text",
    value="I love machine learning!"
)

# -----------------------------
# Expanded emotion keywords
# -----------------------------
positive_words = [
    "love", "great", "amazing", "excellent", "good",
    "awesome", "happy", "best", "fantastic",
    "wonderful", "proud", "impressed", "rewarding",
    "exciting", "smooth", "success", "successful",
    "clean", "modern", "positive", "inspiring",
    "celebrating", "smoothly", "better", "surprised"
]

negative_words = [
    "hate", "bad", "terrible", "awful", "worst",
    "sad", "angry", "horrible", "disappointing",
    "poor", "exhausted", "broken", "chaotic",
    "frustrated", "irritated", "stressful",
    "bugs", "draining", "mess", "quit",
    "overtime", "poorly", "terribly",
    "confused", "ruin", "complained",
    "exhausted", "draining", "chaotic"
]

# -----------------------------
# Split into sentences
# -----------------------------
def split_sentences(text):
    return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]

if st.button("Classify"):

    sentences = split_sentences(text)

    st.subheader("Sentence-Level Analysis")

    results = []

    for i, sentence in enumerate(sentences):

        result = classifier(sentence)[0]
        label = result["label"]
        score = result["score"]

        words = re.findall(r"\b\w+\b", sentence.lower())

        pos = [w for w in words if w in positive_words]
        neg = [w for w in words if w in negative_words]

        st.markdown(f"### Sentence {i+1}")
        st.write(sentence)

        st.write(f"**Model Sentiment:** {label} ({score:.2f})")

        if pos:
            st.success(f"Positive signals: {', '.join(pos)}")

        if neg:
            st.error(f"Negative signals: {', '.join(neg)}")

        # explanation
        if label == "POSITIVE":
            if pos and neg:
                st.info("Mixed emotions detected but positive tone dominates.")
            elif pos:
                st.info("Positive emotional words influenced prediction.")
            else:
                st.info("General positive tone detected.")

        else:
            if pos and neg:
                st.warning("Mixed emotions detected but negative tone dominates.")
            elif neg:
                st.warning("Negative emotional words influenced prediction.")
            else:
                st.warning("General negative tone detected.")

        results.append(label)

    # -----------------------------
    # Overall sentiment logic
    # -----------------------------
    st.subheader("Overall Result")

    pos_count = results.count("POSITIVE")
    neg_count = results.count("NEGATIVE")

    if pos_count > neg_count:
        st.success("Overall Sentiment: POSITIVE (majority of sentences positive)")
    elif neg_count > pos_count:
        st.error("Overall Sentiment: NEGATIVE (majority of sentences negative)")
    else:
        st.warning("Overall Sentiment: MIXED (balanced emotions detected)")

    st.write("Raw output:", results)
