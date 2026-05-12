import streamlit as st
from transformers import pipeline

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():

    sentiment = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    emotion = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

    return sentiment, emotion


sentiment_model, emotion_model = load_models()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Custom HF Sentiment + Emotion AI")

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
    # SENTIMENT
    # -------------------------
    sentiment = sentiment_model(text)[0]

    st.subheader("📊 Sentiment")
    st.success(f"{sentiment['label']} ({sentiment['score']:.2f})")

    # -------------------------
    # EMOTIONS
    # -------------------------
    emotions = emotion_model(text)[0]

    st.subheader("🎭 Detected Emotions")

    for e in emotions:
        st.write(f"{e['label']} → {e['score']:.2f}")

    # -------------------------
    # SIMPLE EXPLANATION LOGIC
    # -------------------------
    st.subheader("💡 Why this result?")

    top_emotions = sorted(emotions, key=lambda x: x["score"], reverse=True)[:3]

    explanation = "The model detected strong emotional signals: "

    explanation += ", ".join([f"{e['label']}" for e in top_emotions])

    explanation += f". Overall sentiment is {sentiment['label']} because these emotions dominate the text."

    st.write(explanation)
