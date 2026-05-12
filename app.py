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
st.title("🧠 Advanced Sentiment Analysis App")

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
    # Normalize output
    # -----------------------------
    if isinstance(raw_output, dict):
        scores = [raw_output]

    elif isinstance(raw_output, list) and isinstance(raw_output[0], list):
        scores = raw_output[0]

    else:
        scores = raw_output

    # -----------------------------
    # Sort scores
    # -----------------------------
    scores = sorted(scores, key=lambda x: float(x["score"]), reverse=True)

    top1 = scores[0]
    top2 = scores[1]

    label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE"
    }

    top_label = label_map.get(top1["label"], top1["label"])
    top_score = float(top1["score"])

    second_score = float(top2["score"])

    # -----------------------------
    # MIXED detection logic
    # -----------------------------
    if abs(top_score - second_score) < 0.15:
        final_label = "MIXED"
    else:
        final_label = top_label

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("📊 Prediction")

    if final_label == "POSITIVE":
        st.success(f"POSITIVE ({top_score:.2f})")

    elif final_label == "NEGATIVE":
        st.error(f"NEGATIVE ({top_score:.2f})")

    elif final_label == "NEUTRAL":
        st.info(f"NEUTRAL ({top_score:.2f})")

    else:
        st.warning(f"MIXED SENTIMENT ({top_score:.2f} vs {second_score:.2f})")

    # -----------------------------
    # Explanation system
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

    if final_label == "MIXED":
        st.info("The model is uncertain — positive and negative signals are very close.")

    elif final_label == "POSITIVE":
        st.info("Overall sentiment is positive.")

    elif final_label == "NEGATIVE":
        st.warning("Overall sentiment is negative.")

    else:
        st.info("Neutral sentiment detected.")

    # -----------------------------
    # Show cues
    # -----------------------------
    if pos_found:
        st.write("🟢 Positive cues:", pos_found)

    if neg_found:
        st.write("🔴 Negative cues:", neg_found)

    # -----------------------------
    # Raw output
    # -----------------------------
    st.subheader("🔍 Raw Model Output")
    st.write(scores)
