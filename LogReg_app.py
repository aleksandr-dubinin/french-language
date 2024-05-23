import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# Load your pre-trained logistic regression model and label encoder
model_filename = 'best_logistic_regression_model.joblib'
label_encoder_filename = 'label_encoder.joblib'

model = joblib.load(model_filename)
label_encoder = joblib.load(label_encoder_filename)

# Predefined article links based on difficulty level and interests
article_links = {
    "A1": {
        "Culture": "https://lingua.com/french/reading/fetes/",
        "Education": "https://lingua.com/french/reading/nombres/",
        "Finance": "https://example.com/a1-finance",
        "Food": "https://example.com/a1-food",
        "Health": "https://example.com/a1-health",
        "Politics": "https://example.com/a1-politics",
        "Science": "https://example.com/a1-science",
        "Sports": "https://example.com/a1-sports",
        "Technology": "https://example.com/a1-technology",
        "Travel": "https://lingua.com/french/reading/voyage/",
    },
    "A2": {
        "Culture": "https://lingua.com/french/reading/fetes/",
        "Education": "https://lingua.com/french/reading/nombres/",
        "Finance": "https://example.com/a1-finance",
        "Food": "https://example.com/a1-food",
        "Health": "https://example.com/a1-health",
        "Politics": "https://example.com/a1-politics",
        "Science": "https://example.com/a1-science",
        "Sports": "https://example.com/a1-sports",
        "Technology": "https://example.com/a1-technology",
        "Travel": "https://lingua.com/french/reading/voyage/",
    },
    "B1": {
        "Culture": "https://lingua.com/french/reading/paques/",
        "Education": "https://progress.lawlessfrench.com/learn/listening/naissance-de-la-langue-francaise",
        "Finance": "https://example.com/a1-finance",
        "Food": "https://example.com/a1-food",
        "Health": "https://example.com/a1-health",
        "Politics": "https://example.com/a1-politics",
        "Science": "https://example.com/a1-science",
        "Sports": "https://example.com/a1-sports",
        "Technology": "https://example.com/a1-technology",
        "Travel": "https://lingua.com/french/reading/marseille/",
    },
    "B2": {
        "Culture": "https://french.kwiziq.com/learn/reading/coco-chanel-portraits-francais",
        "Education": "https://www.lawlessfrench.com/reading/emile-ou-de-leducation/",
        "Finance": "https://example.com/a1-finance",
        "Food": "https://example.com/a1-food",
        "Health": "https://example.com/a1-health",
        "Politics": "https://example.com/a1-politics",
        "Science": "https://example.com/a1-science",
        "Sports": "https://french.kwiziq.com/learn/reading/coupe-du-monde-1998",
        "Technology": "https://example.com/a1-technology",
        "Travel": "https://lingua.com/french/reading/voyage/",
    },
    "C1": {
        "Culture": "https://french.kwiziq.com/learn/reading/coco-chanel-portraits-francais",
        "Education": "https://global-exam.com/blog/en/dalf-c2-reading-writing-section/",
        "Finance": "https://example.com/a1-finance",
        "Food": "https://example.com/a1-food",
        "Health": "https://example.com/a1-health",
        "Politics": "https://example.com/a1-politics",
        "Science": "https://example.com/a1-science",
        "Sports": "https://example.com/a1-sports",
        "Technology": "https://example.com/a1-technology",
        "Travel": "https://lingua.com/french/reading/voyage/",
    },
    "C2": {
        "Culture": "https://french.kwiziq.com/learn/reading/coco-chanel-portraits-francais",
        "Education": "https://global-exam.com/blog/en/dalf-c2-reading-writing-section/",
        "Finance": "https://example.com/a1-finance",
        "Food": "https://example.com/a1-food",
        "Health": "https://example.com/a1-health",
        "Politics": "https://example.com/a1-politics",
        "Science": "https://example.com/a1-science",
        "Sports": "https://example.com/a1-sports",
        "Technology": "https://example.com/a1-technology",
        "Travel": "https://lingua.com/french/reading/voyage/",
    },
    # Add more difficulty levels and corresponding links
}

def predict_difficulty(sentence):
    # Transform the sentence using the TF-IDF vectorizer in the pipeline
    prediction = model.predict([sentence])
    difficulty = label_encoder.inverse_transform(prediction)
    return difficulty[0]

# Custom background set-up with enhanced readability
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            color: #000000;
        }}
        .css-2trqyj {{
            padding: 5px;
            background-color: rgba(255, 255, 255, 0.8);
            border: 1px solid #f5f5dc;
        }}
        .css-2trqyj textarea {{
            background-color: #ffffff;
            border: 1px solid #f5f5dc;
        }}
        .stTextInput {{
            background-color: #ffffff;
            border: 1px solid #f5f5dc;
            padding: 10px;
            border-radius: 5px;
        }}
        .stButton {{
            margin-top: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background to a Paris view
set_background("https://media.istockphoto.com/id/1369337114/vector/france-flag-flag-illustration.jpg?s=612x612&w=0&k=20&c=ibpaq2tnkYL5D-MDaBR9HSKzwzQplrwbR1GONThTWgI=")

# Set up the Streamlit interface
st.title('French Sentence Difficulty Classifier')
sentence = st.text_input("Enter a sentence in French:", key="sentence_input", help="Type a French sentence here.")

if sentence:
    interests = ["Sports", "Technology", "Travel", "Food", "Health", "Science", "Education", "Culture", "Politics", "Finance"]
    user_interest = st.selectbox("Choose your interest:", interests)
    st.markdown("</div>", unsafe_allow_html=True)

    if user_interest:
        difficulty = predict_difficulty(sentence)
        st.write(f"<div style='background-color:#ffffff; border:1px solid #f5f5dc; padding:10px; border-radius:5px;'>Predicted difficulty level of the sentence: {difficulty}</div>", unsafe_allow_html=True)

        if user_interest in article_links[difficulty]:
            article_link = article_links[difficulty][user_interest]
            st.write(f"<div style='background-color:#ffffff; border:1px solid #f5f5dc; padding:10px; border-radius:5px;'>Here is an article for you: <a href='{article_link}' target='_blank'>Read here</a></div>", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("")
# Newsletter subscription section at the bottom
st.markdown("---")  # Horizontal line to separate sections
st.markdown("<div class='stNewsletter'>", unsafe_allow_html=True)
st.write("If you like the app, subscribe to our newsletter!")
email = st.text_input("Enter your email:", key="email_input")
if st.button("Subscribe"):
    st.write("Thank you for subscribing!")
    st.button("Subscribe", disabled=True)
