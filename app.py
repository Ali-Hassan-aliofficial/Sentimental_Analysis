import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------------
# Sentiment model (safe)
# -----------------------------
@st.cache_resource
def load_sentiment():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# -----------------------------
# FLAN-T5 LOADER (FIXED - NO PIPELINE TASK)
# -----------------------------
@st.cache_resource
def load_explainer():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

sentiment_model = load_sentiment()
tokenizer, explainer_model = load_explainer()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 AI Sentiment Analyzer (FIXED FINAL)")

text = st.text_area(
    "Enter text",
    value="I love the product but the experience was frustrating and confusing."
)

# -----------------------------
# RUN
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    # -------------------------
    # 1. Sentiment
    # -------------------------
    result = sentiment_model(text)[0]

    label = result["label"]
    score = result["score"]

    st.subheader("📊 Sentiment")
    st.success(f"{label} ({score:.2f})")

    # -------------------------
    # 2. Explanation (NO PIPELINE)
    # -------------------------
    prompt = f"""
Explain sentiment in simple terms.

Text:
{text}

Sentiment:
{label}

Give explanation and key emotional words.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = explainer_model.generate(
        **inputs,
        max_new_tokens=150
    )

    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("💡 Explanation")
    st.write(explanation)
