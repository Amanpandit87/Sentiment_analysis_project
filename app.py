import streamlit as st
import pickle as pkl
import re

# Load the model and vectorizer
model = pkl.load(open("sentiment_analysis_model.pkl", "rb"))
tfidf = pkl.load(open("sentiment_analysis_tfidf.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub("http\S+\s", " ", text)
    text = re.sub("RT|CC", " ", text)
    text = re.sub("#\S+\s", " ", text)
    text = re.sub("@\S+", " ", text)
    text = re.sub("[%s]" % re.escape("""!"#&%&"()*+,-./:;<=>?@[\]^_'{|}~"""), " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

# Sentiment label decoder
def decode_sentiment(label):
    if label == 0:
        return "negative"
    elif label == 1:
        return "neutral"
    elif label == 2:
        return "positive"
    else:
        return "unknown"

# Streamlit GUI
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("🧠 Sentiment Analysis Web App")
st.write("Enter a sentence or tweet and find out the sentiment!")

user_input = st.text_area("Enter text:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = decode_sentiment(prediction)

        st.success(f"Predicted Sentiment: **{sentiment.capitalize()}**")
