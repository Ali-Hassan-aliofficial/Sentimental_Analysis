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

    # FIX: use text-generation instead of text2text-generation
    explainer = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=80,
        do_sample=False
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
# Run
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    # -----------------------------
    # Sentiment
    # -----------------------------
    result = sentiment_model(text)[0]

    label = result["label"]
    confidence = result["score"]

    # -----------------------------
    # Explanation prompt
    # -----------------------------
    prompt = (
        f"Explain why this text is {label.lower()} sentiment in simple terms:\n\n"
        f"{text}\n\n"
        f"Answer:"
    )

    explanation = explainer(prompt)[0]["generated_text"]

    # -----------------------------
    # UI output
    # -----------------------------
    st.subheader("📊 Result")
    st.success(f"Sentiment: {label}")
    st.info(f"Confidence: {confidence}")

    st.subheader("💡 Explanation")
    st.write(explanation)
