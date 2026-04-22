import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load the NLP model and tf-idf vectorizer
model = joblib.load('nlp_model.pkl')
tfidf = joblib.load('tfidf.pkl')

# Initialize the NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to clean and predict sentiment
def predict_sentiment(review):
    if not review.strip():
        return "Input is empty. Please enter some text."
    
    # Clean and preprocess the input review
    cleaned_review = re.sub(r'[^\w\s]', '', review)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in stop_words]
    stemmed_review = [stemmer.stem(word) for word in filtered_review]
    
    # Transform the preprocessed review using the saved TF-IDF vectorizer
    tfidf_review = tfidf.transform([' '.join(stemmed_review)])
    
    # Convert the sparse matrix to a dense array (required by Keras model)
    tfidf_review_dense = tfidf_review.toarray()

    # Predict the sentiment using the loaded model
    sentiment_prediction = model.predict(tfidf_review_dense)
    
    # Assuming binary classification with sigmoid output
    if sentiment_prediction[0][1] > 0.5:
        return "😃Positive"
    else:
        return "😫Negative"


# Streamlit UI
st.title("Sentiment Analysis")
review_to_predict = st.text_area("Enter your text here...")

if st.button("Predict Sentiment"):
    sentiment = predict_sentiment(review_to_predict)
    st.write("Predicted sentiment: ", sentiment)
