import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Load GPT-2
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 GPT-2 Sentiment Explainer (Demo Version)")

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
You are a sentiment analysis AI.

Text:
{text}

Step 1: Identify sentiment (Positive, Negative, or Mixed)
Step 2: Explain why
Step 3: List emotional words

Answer:
"""

    with st.spinner("GPT-2 is generating response..."):

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("🤖 GPT-2 Response")
    st.write(result)
