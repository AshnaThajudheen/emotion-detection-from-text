# Emotion Detection from Text

An NLP course project that classifies English text into six emotion
categories using TF-IDF features and machine learning.

## Emotions
- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

## Models
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

The models are compared using validation macro F1-score. The selected
model is evaluated on a separate test set.

## Methodology
1. Load the labeled emotion dataset.
2. Convert text into TF-IDF features using unigrams and bigrams.
3. Train and compare two machine-learning classifiers.
4. Evaluate the selected model on unseen test data.
5. Generate a classification report and confusion matrix.

## Project structure
- `Emotion_Detection_NLP.ipynb`: training and evaluation notebook
- `models/`: saved model and TF-IDF vectorizer
- `results/`: evaluation metrics, predictions and visualizations

## Limitations
The classifier always predicts one of the six available emotions.
Neutral text may be assigned an emotion even when no clear emotion exists.
Predictions are not guaranteed to be correct.

## Dataset
Hugging Face: dair-ai/emotion

## Run
Open the notebook in Google Colab and run the cells in order.

## Live Demo

Try the deployed application here:

**[Emotion Detection from Text — Open Live App](https://emotion-detection-from-text-nlp-course.streamlit.app/)**

Enter an English sentence and let the model predict its dominant emotion: Sadness, Joy, Love, Anger, Fear, or Surprise.
