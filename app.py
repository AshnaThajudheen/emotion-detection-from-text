
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Emotion Detection from Text",
    page_icon="💭",
    layout="centered"
)

MODEL_DIR = Path(__file__).parent / "models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "emotion_model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("💭 Emotion Detection from Text")
st.write(
    "Enter an English sentence and let the NLP model "
    "predict its dominant emotion."
)

emotion_icons = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲"
}

text = st.text_area(
    "Enter your text",
    placeholder="Example: I am so happy about my achievement!",
    height=130
)

if st.button("Detect Emotion", type="primary"):
    if not text.strip():
        st.warning("Please enter a sentence first.")
    else:
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]

        # The dataset uses integer labels 0 through 5.
        # Map them to the dataset's original emotion names.
        label_names = [
            "sadness", "joy", "love",
            "anger", "fear", "surprise"
        ]
        emotion = label_names[int(prediction)]

        st.subheader("Predicted Emotion")
        st.markdown(
            f"## {emotion_icons.get(emotion, '💭')} "
            f"{emotion.title()}"
        )

        st.caption(
            "This is a machine-learning prediction, "
            "not a definitive interpretation of emotion."
        )

st.divider()
st.caption("NLP Course Project | TF-IDF + Linear SVM")
