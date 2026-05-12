import streamlit as st
from transformers import pipeline

# -----------------------------
# Load model safely (NO task errors)
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Sentiment Analysis AI (Fixed Version)")

text = st.text_area(
    "Enter text",
    value="I love the product but the experience was very frustrating and confusing."
)

# -----------------------------
# Run
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    prompt = f"""
Analyze sentiment.

Text:
{text}

Return:
1. Sentiment (Positive / Negative / Neutral / Mixed)
2. Short explanation
3. Key emotional words
"""

    with st.spinner("AI thinking..."):

        result = model(
            prompt,
            max_new_tokens=150,
            do_sample=False,
            return_full_text=False
        )[0]["generated_text"]

    st.subheader("🤖 Result")
    st.write(result)
