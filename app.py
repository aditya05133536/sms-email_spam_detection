import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the pre-trained vectorizer and model
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    st.error("Vectorizer file not found. Ensure 'vectorizer.pkl' is in the correct directory.")

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Ensure 'model.pkl' is in the correct directory.")

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:
        # 1. Preprocess the input message
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the transformed message
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict using the pre-trained model
        try:
            result = model.predict(vector_input)[0]
        except AttributeError as e:
            st.error(f"Model not fitted: {e}")
            result = None

        # 4. Display the result
        if result == 1:
            st.header("Spam")
        elif result == 0:
            st.header("Not Spam")
        else:
            st.error("Unable to make a prediction.")
    else:
        st.warning("Please enter a message to classify.")
