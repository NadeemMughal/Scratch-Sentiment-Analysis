# Importing Necessary Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Define a function to clean the tweet
def clean_tweet(tweet):
    # Remove URLs, mentions, and hashtags
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@[^\s]+[\s]?", "", tweet)
    tweet = re.sub(r"#([^\s]+[\s]?)+", "", tweet)

    # Remove stopwords and stem the remaining words
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # Tokenize words explicitly in English
    words = nltk.word_tokenize(tweet, language='english')

    # Stem and remove stopwords
    words = [stemmer.stem(word) for word in words if word not in stopwords_english]
    return " ".join(words)

# Define the Streamlit app
def app():
    # Set the page title, layout, and page icon
    st.set_page_config(page_title='Sentiment Analysis App', page_icon=':smiley:', layout='centered')

    # Create a header section with a title and styling
    st.markdown("""
        <div style="background-color:#4CAF50;padding:10px;border-radius:10px;">
            <h2 style="color:white;text-align:center;">Welcome to the Sentiment Analysis App</h2>
        </div>
    """, unsafe_allow_html=True)

    # Add a warning about model limitations
    st.warning("⚠️ **Please Note:** This model is trained on a limited dataset scraped from Twitter in real-time. "
               "As a result, its robustness may not meet all use cases. ")

    st.write('### Enter a tweet below to predict its sentiment.')

    # Add a text input for user input
    tweet_input = st.text_input('Enter a tweet:')

    # Add a button for analysis
    analyze_button = st.button('Analyze Sentiment')

    # Check if the user has entered a tweet and clicked the button
    if tweet_input and analyze_button:
        # Clean the tweet
        cleaned_tweet = clean_tweet(tweet_input)

        # Vectorize the cleaned tweet using the pre-trained vectorizer
        tweet_vectorized = vectorizer.transform([cleaned_tweet])

        # Predict the sentiment of the tweet using the pre-trained model
        sentiment = model.predict(tweet_vectorized)[0]

        # Display the predicted sentiment with icons
        if sentiment == 'positive':
            st.markdown('<h3 style="color:green;">The sentiment is positive :😊:</h3>', unsafe_allow_html=True)
        elif sentiment == 'negative':
            st.markdown('<h3 style="color:red;">The sentiment is negative :😞:</h3>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 style="color:gray;">The sentiment is neutral :😐:</h3>', unsafe_allow_html=True)

    # Add a footer with social media links (LinkedIn and GitHub)
    st.markdown("""
        <hr>
        <div style="text-align:center;">
            <p>Created by <b>Muhammad Nadeem</b></p>
            <p>
                <a href="https://www.linkedin.com/in/muhammad-nadeem-5a1517242/" target="_blank">LinkedIn</a> | 
                <a href="https://github.com/NadeemMughal" target="_blank">GitHub</a>
            </p>
        </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    app()
