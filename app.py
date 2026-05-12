import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------------
# Load model safely
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 AI Sentiment Explainer (Stable & Clean)")

text = st.text_area(
    "Enter your text",
    value="I love the product but the experience was frustrating, messy, and confusing."
)

# -----------------------------
# Run analysis
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    prompt = f"""
    You are a sentiment analysis expert.

    Task:
    1. Identify sentiment (Positive / Negative / Neutral / Mixed)
    2. Explain why in simple terms
    3. Mention key emotional words from the text

    Text:
    {text}

    Answer:
    """

    with st.spinner("AI is analyzing..."):

        result = model(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("🤖 AI Analysis")
    st.write(result)
