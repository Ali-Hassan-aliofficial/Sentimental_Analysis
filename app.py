import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )

classifier = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Sentiment Analysis App (Stable Version)")

text = st.text_area(
    "Enter your text",
    value="I love this product but the experience was frustrating and slow"
)

# -----------------------------
# Run analysis
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analyzing..."):

        raw_output = classifier(text)

    # -----------------------------
    # FIX OUTPUT STRUCTURE
    # -----------------------------
    scores = raw_output[0]  # IMPORTANT FIX

    best = max(scores, key=lambda x: x["score"])

    label = best["label"]
    score = best["score"]

    # -----------------------------
    # Label mapping
    # -----------------------------
    label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE"
    }

    pretty_label = label_map.get(label, label)

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("📊 Prediction")
    st.success(f"{pretty_label} ({score:.2f})")

    # -----------------------------
    # Simple explanation
    # -----------------------------
    st.subheader("💡 Explanation")

    words = text.lower().split()

    positive_words = ["good", "love", "great", "awesome", "excellent", "happy", "smooth"]
    negative_words = ["bad", "hate", "terrible", "slow", "frustrating", "worst", "chaotic"]

    pos_found = [w for w in words if w in positive_words]
    neg_found = [w for w in words if w in negative_words]

    if pretty_label == "POSITIVE":
        st.info("The model predicts POSITIVE sentiment based on overall tone.")

    elif pretty_label == "NEGATIVE":
        st.warning("The model predicts NEGATIVE sentiment based on negative cues.")

    else:
        st.info("The model predicts NEUTRAL / MIXED sentiment.")

    if pos_found:
        st.write("🟢 Positive cues detected:", pos_found)

    if neg_found:
        st.write("🔴 Negative cues detected:", neg_found)

    # -----------------------------
    # Raw output
    # -----------------------------
    st.subheader("🔍 Raw Model Output")
    st.write(scores)
