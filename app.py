import streamlit as st
import joblib
import PyPDF2
import json
import time

from utils import preprocess_text  # Custom text preprocessing function

# =========================
# Configuration & Settings
# =========================
st.set_page_config(
    page_title="AI Resume Category Predictor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
CATEGORY_INFO_FILE = "category_info.json"

# =========================
# Caching for performance
# =========================
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"🚨 {e}")
        st.stop()

@st.cache_resource
def load_category_info(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"🚨 Failed to load category info: {e}")
        st.stop()

# =========================
# Helper Functions
# =========================
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        st.error(f"❌ PDF extraction error: {e}")
        return ""

def get_top_n_keywords(vectorizer, text, n=10):
    if not hasattr(vectorizer, 'get_feature_names_out'):
        return []
    features = vectorizer.transform([text]).toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    scored_features = sorted(
        [(score, feature_names[i]) for i, score in enumerate(features) if score > 0],
        key=lambda x: x[0],
        reverse=True
    )
    return [feature for _, feature in scored_features[:n]]

def get_resume_strength(text):
    score = 0
    feedback = []
    word_count = len(text.split())
    lower_text = text.lower()

    # Length
    if word_count > 300:
        score += 1
        feedback.append("✔️ Sufficient content length.")
    else:
        feedback.append("💡 Consider expanding your resume with more details.")

    sections = {
        "experience": ["experience", "work history"],
        "education": ["education"],
        "skills": ["skills", "technical skills"],
        "projects": ["project", "portfolio"],
        "contact": ["email", "phone", "linkedin", "contact"]
    }

    for section, keywords in sections.items():
        if any(kw in lower_text for kw in keywords):
            score += 1
            feedback.append(f"✔️ '{section.capitalize()}' section detected.")
        else:
            feedback.append(f"🔍 Consider adding a clear '{section.capitalize()}' section.")

    rating = "💪 Excellent" if score >= 4 else "👍 Good" if score >= 2 else "👇 Needs Work"
    return rating, feedback

# =========================
# Load resources
# =========================
model, vectorizer = load_model_and_vectorizer()
CATEGORY_INFO = load_category_info(CATEGORY_INFO_FILE)

# =========================
# UI Design
# =========================
st.title("📄 AI Resume Category Predictor")

st.markdown("""
Upload your **resume** in PDF or TXT format. Our AI will:
- Predict the **most relevant job category**
- Provide **personalized tips** to improve your resume
- Suggest **keywords** and analyze structure
""")

st.markdown("---")
uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])

resume_text = ""
if uploaded_file:
    with st.spinner("🔍 Extracting text..."):
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")

    if resume_text.strip():
        with st.expander("📄 Preview Extracted Resume Text"):
            st.text_area("Resume Content", resume_text[:3000], height=250)

        if st.button("🔮 Predict Category"):
            with st.spinner("🧠 Analyzing..."):
                cleaned_text = preprocess_text(resume_text)
                if not cleaned_text:
                    st.warning("Resume appears empty after preprocessing.")
                else:
                    try:
                        vectorized = vectorizer.transform([cleaned_text])
                        predicted = model.predict(vectorized)[0]

                        st.success(f"🎯 **Predicted Category:** {predicted}")
                        st.markdown("---")

                        # Category Tips
                        if predicted in CATEGORY_INFO:
                            info = CATEGORY_INFO[predicted]
                            st.subheader("📌 Category Insights")
                            st.info(info.get("description", ""))
                            st.markdown("#### ✨ Tips:")
                            for tip in info.get("tips", []):
                                st.markdown(f"- {tip}")
                        else:
                            st.warning("No detailed tips available for this category.")

                        # Keywords
                        st.subheader("🔑 Top Keywords")
                        keywords = get_top_n_keywords(vectorizer, cleaned_text)
                        if keywords:
                            st.markdown(" ".join([f"`{kw}`" for kw in keywords]))
                        else:
                            st.info("Could not extract meaningful keywords.")

                        # Resume Strength
                        st.subheader("📈 Resume Strength")
                        strength, feedback = get_resume_strength(cleaned_text)
                        st.metric("Assessment", strength)
                        with st.expander("💬 Detailed Feedback"):
                            for f in feedback:
                                st.markdown(f"- {f}")

                        # Download
                        st.download_button(
                            label="⬇️ Download Preprocessed Resume Text",
                            data=cleaned_text.encode(),
                            file_name="cleaned_resume.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
    else:
        st.warning("The file appears to be empty or unsupported.")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **machine learning model** to classify resumes into job categories and suggest improvements.
    """)
    with st.expander("How it Works"):
        st.markdown("""
        1. Extracts resume text (PDF/TXT)  
        2. Preprocesses using `utils.py`  
        3. Vectorizes with TF-IDF  
        4. Predicts with `model.pkl`  
        5. Matches category info from `category_info.json`
        """)
    st.caption("Built with ❤️ using Streamlit & Scikit-learn.")
