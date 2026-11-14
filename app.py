
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# AI vs Human Academic Writing Classification
# Streamlit Deployment Application
# =========================================================

st.set_page_config(
    page_title="AI vs Human Writing Classifier",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "final_random_forest_model.pkl"

# -------------------- Custom Styling --------------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #142d4e, #176b78);
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 34px;
}
.hero p {
    color: #e6f1f5;
    font-size: 16px;
}
.section-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e6eaf0;
    margin-bottom: 15px;
}
.result {
    padding: 25px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e2e8f0;
    text-align: center;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e6eaf0;
}
</style>
""", unsafe_allow_html=True)


# -------------------- Load Trained Model --------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# -------------------- Header --------------------

st.markdown("""
<div class="hero">
    <h1>✍️ AI vs Human Academic Writing</h1>
    <p>Machine Learning Based Academic Writing Classification</p>
</div>
""", unsafe_allow_html=True)


# -------------------- Model Availability --------------------

if not MODEL_PATH.exists():
    st.error(
        "Trained model file not found. Please place "
        "'final_random_forest_model.pkl' in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()


# -------------------- Sidebar --------------------

with st.sidebar:
    st.title("Project Dashboard")
    st.markdown("---")
    st.write("**Project**")
    st.write("AI vs Human Academic Writing Classification")
    st.write("**Selected Model**")
    st.write("Random Forest Classifier")
    st.write("**Input Features**")
    st.write("18 Features")
    st.markdown("---")
    st.caption("Foundation of Artificial Intelligence")
    st.caption("Individual Machine Learning Project")


# -------------------- Introduction --------------------

st.markdown("""
### Welcome

Enter the academic submission characteristics below to obtain
a prediction from the trained machine learning model.
""")

st.info(
    "This application uses structured submission characteristics. "
    "It does not analyse pasted essay text."
)


# -------------------- Input Form --------------------

with st.form("prediction_form"):

    st.markdown("### 1. Submission Information")

    col1, col2 = st.columns(2)

    with col1:
        submission_type = st.selectbox(
            "Submission Type",
            ["Essay", "Research_Paper", "Lab_Report", "Literature_Review"]
        )

        academic_level = st.selectbox(
            "Academic Level",
            ["Undergraduate", "Postgraduate", "High_School"]
        )

    with col2:
        primary_language = st.selectbox(
            "Primary Language",
            ["Native", "Non_Native"]
        )

        font_family = st.selectbox(
            "Font Family",
            ["Calibri", "Arial", "Times New Roman", "Helvetica"]
        )

    st.markdown("---")
    st.markdown("### 2. Numerical Characteristics")

    c1, c2, c3 = st.columns(3)

    with c1:
        word_count = st.number_input(
            "Word Count", min_value=0, value=500
        )

        avg_sentence_length = st.number_input(
            "Average Sentence Length",
            min_value=0.0, value=15.0
        )

        grammar_errors = st.number_input(
            "Grammar Errors", min_value=0, value=3
        )

        vocabulary_richness = st.number_input(
            "Vocabulary Richness",
            min_value=0.0, value=0.60, format="%.3f"
        )

        passive_voice_ratio = st.number_input(
            "Passive Voice Ratio",
            min_value=0.0, value=0.20, format="%.3f"
        )

    with c2:
        reading_level = st.number_input(
            "Reading Level", min_value=0.0, value=10.0
        )

        editing_time = st.number_input(
            "Editing Time", min_value=0.0, value=30.0
        )

        revision_count = st.number_input(
            "Revision Count", min_value=0, value=3
        )

        typing_speed = st.number_input(
            "Typing Speed", min_value=0.0, value=40.0
        )

        plagiarism_score = st.number_input(
            "Plagiarism Score", min_value=0.0, value=5.0
        )

    with c3:
        citation_count = st.number_input(
            "Citation Count", min_value=0, value=5
        )

        punctuation_density = st.number_input(
            "Punctuation Density",
            min_value=0.0, value=0.05, format="%.3f"
        )

        sentence_complexity = st.number_input(
            "Sentence Complexity",
            min_value=0.0, value=0.50, format="%.3f"
        )

        submission_hour = st.number_input(
            "Submission Hour (0–23)",
            min_value=0, max_value=23, value=14
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 Predict Writing Category",
        use_container_width=True,
        type="primary"
    )


# -------------------- Prediction --------------------

if submitted:

    input_data = pd.DataFrame([{
        "Word_Count": word_count,
        "Average_Sentence_Length": avg_sentence_length,
        "Grammar_Errors": grammar_errors,
        "Vocabulary_Richness": vocabulary_richness,
        "Passive_Voice_Ratio": passive_voice_ratio,
        "Reading_Level": reading_level,
        "Editing_Time": editing_time,
        "Revision_Count": revision_count,
        "Typing_Speed": typing_speed,
        "Plagiarism_Score": plagiarism_score,
        "Citation_Count": citation_count,
        "Punctuation_Density": punctuation_density,
        "Sentence_Complexity": sentence_complexity,
        "Submission_Hour": submission_hour,
        "Submission_Type": submission_type,
        "Academic_Level": academic_level,
        "Primary_Language": primary_language,
        "Font_Family": font_family
    }])

    try:
        prediction = int(model.predict(input_data)[0])

        st.markdown("---")
        st.markdown("### 3. Prediction Results")

        if prediction == 1:
            st.markdown("""
            <div class="result">
                <h2 style="color:#b45309;">AI-Assisted Writing</h2>
                <p>The model classified this submission as AI-assisted.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result">
                <h2 style="color:#087f5b;">Human-Written</h2>
                <p>The model classified this submission as human-written.</p>
            </div>
            """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            ai_probability = float(
                probabilities[classes.index(1)]
            )
            human_probability = float(
                probabilities[classes.index(0)]
            )

            st.markdown("#### Prediction Probabilities")

            p1, p2 = st.columns(2)

            p1.metric(
                "Human-Written",
                f"{human_probability:.1%}"
            )

            p2.metric(
                "AI-Assisted",
                f"{ai_probability:.1%}"
            )

            probability_data = pd.DataFrame({
                "Estimated Probability": [
                    human_probability,
                    ai_probability
                ]
            }, index=["Human-Written", "AI-Assisted"])

            st.bar_chart(probability_data)

        st.caption(
            "The prediction is a machine learning estimate, "
            "not definitive proof of authorship."
        )

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.exception(error)


# -------------------- Project Information --------------------

st.markdown("---")

with st.expander("About This Project"):

    st.write("""
    This project applies machine learning techniques to classify
    academic writing as human-written or AI-assisted.

    The project compares four machine learning algorithms:

    - Dummy Classifier
    - Logistic Regression
    - Decision Tree
    - Random Forest

    The selected Random Forest model uses the preprocessing
    pipeline developed during the project.

    Numerical features are processed using median imputation
    and RobustScaler. Categorical features are processed using
    most-frequent imputation and One-Hot Encoding.
    """)

st.markdown("---")

st.caption(
    "Foundation of AI | Individual Machine Learning Project"
)
