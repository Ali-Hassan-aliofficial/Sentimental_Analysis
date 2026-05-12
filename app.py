import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_classifier():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_classifier()

st.title("Sentiment Analysis App")

st.write(
    "Enter a sentence below and click **Classify** to predict sentiment."
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

                st.write(result)
                st.write(text)

            else:
                st.write(result)
