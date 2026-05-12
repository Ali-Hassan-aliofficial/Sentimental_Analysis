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

# Simple keyword lists
positive_words = [
    "love", "great", "amazing", "excellent",
    "good", "awesome", "happy", "best",
    "fantastic", "wonderful"
]

negative_words = [
    "hate", "bad", "terrible", "awful",
    "worst", "sad", "angry", "horrible",
    "disappointing", "poor"
]

st.title("Sentiment Analysis App")

st.write(
    "Enter a sentence below and click classify."
)

text = st.text_area(
    "Your sentence",
    value="I love machine learning!"
)

if st.button("Classify"):

    with st.spinner("Classifying..."):

        try:
            result = classifier(text)

        except Exception as e:
            st.error(f"Model inference failed: {e}")

        else:

            if isinstance(result, list) and result:

                res = result[0]

                label = res.get("label")
                score = res.get("score")

                st.success(f"Sentiment: {label}")
                st.info(f"Confidence Score: {score:.3f}")

                # -----------------------------
                # Word Analysis
                # -----------------------------

                words = re.findall(r"\b\w+\b", text.lower())

                detected_positive = []
                detected_negative = []

                for word in words:

                    if word in positive_words:
                        detected_positive.append(word)

                    if word in negative_words:
                        detected_negative.append(word)

                st.subheader("Detected Important Words")

                if detected_positive:
                    st.write(
                        f"Positive words found: {', '.join(detected_positive)}"
                    )

                if detected_negative:
                    st.write(
                        f"Negative words found: {', '.join(detected_negative)}"
                    )

                # -----------------------------
                # Explanation
                # -----------------------------

                st.subheader("Why Did The Model Predict This?")

                if label == "POSITIVE":

                    if detected_positive:
                        st.write(
                            "The model predicted POSITIVE because it detected "
                            "strong positive words and emotional tone."
                        )

                    else:
                        st.write(
                            "The sentence overall appeared emotionally positive."
                        )

                else:

                    if detected_negative:
                        st.write(
                            "The model predicted NEGATIVE because it detected "
                            "negative emotional words and tone."
                        )

                    else:
                        st.write(
                            "The sentence overall appeared emotionally negative."
                        )

                st.subheader("Raw Model Output")
                st.write(result)

            else:
                st.write(result)
