import streamlit as st
from transformers import pipeline

# -----------------------------
# Load FLAN-T5 model (better reasoning)
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

model = load_model()

st.title("🧠 AI Sentiment Explainer (Better Model)")

text = st.text_area(
    "Enter your text",
    value="I love the product but the experience was frustrating, chaotic, and sometimes really bad."
)

if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    prompt = f"""
    Classify the sentiment of this text and explain why.

    Text:
    {text}

    Answer format:
    Sentiment:
    Explanation:
    """

    with st.spinner("Thinking like an AI..."):

        result = model(prompt, max_length=200)[0]["generated_text"]

    st.subheader("🤖 AI Response")
    st.write(result)
