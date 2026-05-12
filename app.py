import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():

    # Sentiment classifier
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Explanation model
    explainer = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=120,
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
# Analyze
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
    confidence = result["score"]

    # -----------------------------
    # Generate explanation
    # -----------------------------
    prompt = f"""
<|system|>
You are an AI sentiment explanation assistant.

<|user|>
Text: "{text}"

Sentiment: {label}

Explain in 2-3 simple sentences WHY this sentiment was predicted.

<|assistant|>
"""

    output = explainer(prompt)

    explanation = output[0]["generated_text"]

    # Remove prompt from output
    explanation = explanation.replace(prompt, "").strip()

    # -----------------------------
    # Extract emotional words
    # -----------------------------
    words = text.lower().split()

    pos_words = []
    neg_words = []

    for word in words:

        clean_word = word.strip(".,!?()[]{}\"'")

        score = vader.lexicon.get(clean_word)

        if score is not None:

            if score > 0:
                pos_words.append(clean_word)

            elif score < 0:
                neg_words.append(clean_word)

    # Remove duplicates
    pos_words = list(set(pos_words))
    neg_words = list(set(neg_words))

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("📊 Result")

    st.success(f"Sentiment: {label}")
    st.info(f"Confidence: {confidence}")

    st.subheader("💡 Explanation")

    st.write(explanation)

    # -----------------------------
    # Emotional cues
    # -----------------------------
    if pos_words or neg_words:

        st.subheader("🔍 Emotional Signals")

        if pos_words:
            st.write("Positive cues:", pos_words)

        if neg_words:
            st.write("Negative cues:", neg_words)
