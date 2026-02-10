import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Production Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Custom Styling (Red Background)
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Artifacts (Cached)
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)           # remove mentions
    text = re.sub(r"[^a-z\s]", "", text)      # remove punctuation/numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(text: str):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # Optional probability (if supported)
    try:
        probs = model.predict_proba(vectorized)[0]
        confidence = max(probs)
    except Exception:
        confidence = None

    return prediction, confidence

# -----------------------------
# UI
# -----------------------------
st.title("📊 Production-Grade Sentiment Analysis")
st.markdown("""
Analyze emotional tone from text using a professionally deployed Machine Learning model.

**Use cases:**
- Customer feedback monitoring  
- Social media intelligence  
- Product review analysis  
- Brand sentiment tracking
""")

st.divider()

user_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Example: I absolutely love this product — it exceeded my expectations!"
)

col1, col2 = st.columns([1,1])

with col1:
    analyze_btn = st.button("Analyze Sentiment", use_container_width=True)

with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.experimental_rerun()

if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Running inference..."):
            prediction, confidence = predict_sentiment(user_input)

        st.divider()

        # Display result with styling
        if str(prediction).lower() in ["positive", "pos"]:
            st.success(f"✅ **Sentiment:** {prediction}")
        elif str(prediction).lower() in ["negative", "neg"]:
            st.error(f"❌ **Sentiment:** {prediction}")
        else:
            st.info(f"ℹ️ **Sentiment:** {prediction}")

        if confidence is not None:
            st.progress(float(confidence))
            st.caption(f"Model confidence: **{confidence:.2%}**")

st.divider()

# -----------------------------
# Sidebar (Professional Touch)
# -----------------------------
st.sidebar.header("Deployment Info")
st.sidebar.markdown("""
**Model:** TF-IDF + Classical ML  
**Status:** Production Ready  
**Inference:** Real-time  
""")

st.sidebar.header("Best Practices Implemented")
st.sidebar.markdown("""
✔ Cached model loading  
✔ Deterministic preprocessing  
✔ Confidence scoring  
✔ Clean UX  
✔ Failure-safe inference
""")

st.sidebar.header("Author")
st.sidebar.markdown("""
**Abidemi Avoseh**  
Machine Learning Engineer | Data Scientist | AI Engineer
""")

