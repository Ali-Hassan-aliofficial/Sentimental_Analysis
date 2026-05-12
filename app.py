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
st.title("🧠 Sentiment Analysis App (Robust Version)")

text = st.text_area(
    "Enter your text",
    value="I love this product but the experience was frustrating and confusing"
)

# -----------------------------
# Run analysis
# -----------------------------
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analyzing sentiment..."):

        raw_output = classifier(text)

    # -----------------------------
    # SAFE OUTPUT HANDLING
    # -----------------------------
    if isinstance(raw_output, dict):
        scores = [raw_output]

    elif isinstance(raw_output, list) and isinstance(raw_output[0], list):
        scores = raw_output[0]

    else:
        scores = raw_output

    # -----------------------------
    # Find best result safely
    # -----------------------------
    best = max(scores, key=lambda x: float(x["score"]))

    label = best["label"]
    score = float(best["score"])

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
    # Simple AI explanation
    # -----------------------------
    st.subheader("💡 Explanation")

    words = text.lower().split()

    positive_words = [
        "good", "love", "great", "awesome",
        "excellent", "happy", "smooth", "best",
        "amazing", "nice", "positive"
    ]

    negative_words = [
        "bad", "hate", "terrible", "slow",
        "frustrating", "worst", "chaotic",
        "confusing", "poor", "annoying"
    ]

    pos_found = [w for w in words if w in positive_words]
    neg_found = [w for w in words if w in negative_words]

    if pretty_label == "POSITIVE":
        st.info("The model detected overall POSITIVE sentiment in the text.")

    elif pretty_label == "NEGATIVE":
        st.warning("The model detected NEGATIVE sentiment in the text.")

    else:
        st.info("The model detected NEUTRAL or MIXED sentiment.")

    # Show cues
    if pos_found:
        st.write("🟢 Positive cues:", pos_found)

    if neg_found:
        st.write("🔴 Negative cues:", neg_found)

    # -----------------------------
    # Raw output (debug)
    # -----------------------------
    st.subheader("🔍 Raw Model Output")
    st.write(scores)
