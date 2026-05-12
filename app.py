import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Load model safely (NO PIPELINE)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Sentiment Analysis AI (Fully Fixed)")

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

    prompt = f"""
Analyze sentiment.

Text:
{text}

Return:
Sentiment, Explanation, Key words
"""

    with st.spinner("AI is thinking..."):

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("🤖 Result")
    st.write(result)
