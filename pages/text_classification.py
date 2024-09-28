import joblib
import numpy as np
from deep_translator import GoogleTranslator

# Define CustomCountVectorizer and CustomMultinomialNB
class CustomCountVectorizer:
    def __init__(self):
        self.vocabulary_ = {}
        
    def fit(self, X):
        idx = 0
        for text in X:
            for word in text.split():
                if word not in self.vocabulary_:
                    self.vocabulary_[word] = idx
                    idx += 1
                    
    def transform(self, X):
        vectors = np.zeros((len(X), len(self.vocabulary_)))
        for i, text in enumerate(X):
            for word in text.split():
                if word in self.vocabulary_:
                    vectors[i][self.vocabulary_[word]] += 1
        return vectors
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class CustomMultinomialNB:
    def __init__(self):
        self.class_priors = None
        self.feature_likelihoods = None
        self.classes = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.class_priors = np.zeros(n_classes, dtype=np.float64)
        self.feature_likelihoods = np.zeros((n_classes, n_features), dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[idx] = X_c.shape[0] / float(n_samples)
            self.feature_likelihoods[idx, :] = (X_c.sum(axis=0) + 1) / (X_c.sum() + n_features)
            
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.class_priors[idx])
            log_likelihood = np.sum(np.log(self.feature_likelihoods[idx, :]) * x)
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# Load model and vectorizer
model_filename = r'C:\Users\ASUS\Desktop\College\Sem7\DL\M1_DL_Project\DL_MiniProject\language_detection_model.pkl'
vectorizer_filename = r'C:\Users\ASUS\Desktop\College\Sem7\DL\M1_DL_Project\DL_MiniProject\count_vectorizer.pkl'

# Load the trained model and vectorizer
model = joblib.load(model_filename)
cv = joblib.load(vectorizer_filename)

# Streamlit app for text classification and translation
import streamlit as st

st.title("Text Classification and Translation")
st.write("Enter text in any supported language, and the app will detect the language and translate it to English.")

# Add a text area for user input
user_input = st.text_area("Enter text (max 1000 words):", height=300)

# Add a submit button
submit_button = st.button("Submit")

# If the submit button is pressed, perform language detection and translation
if submit_button and user_input:
    # Transform input using the count vectorizer
    data = cv.transform([user_input])
    
    # Predict the language
    predicted_language = model.predict(data)[0]
    st.write(f"Predicted Language: {predicted_language}")
    
    # Translate the text to English
    translated_text = GoogleTranslator(source='auto', target='en').translate(user_input)
    st.write(f"Translated Text: {translated_text}")
