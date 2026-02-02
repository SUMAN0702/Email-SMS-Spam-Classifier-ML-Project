import nltk
import stopwords
import  streamlit as st
import pickle
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Text Preprocessing
def transform_text(text):
    text = text.lower()#Lower case
    text = nltk.word_tokenize(text) #tokenization

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():# isalnum() checks if the string consists of alphanumeric characters only
            y.append(i)

    text = y[:]# copy the list
    y.clear()# clear the list

    # Removing stop words and punctuation
    import string
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]# copy the list
    y.clear()# clear the list

    # Stemming or lemitization- it means reducing a word to its root word like playing, played, plays to play
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()# create an object of PorterStemmer class
    for i in text:
        y.append(ps.stem(i)) # type: ignore


    return " ".join(y)# join the list elements with space


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_input('Enter the message')

# 1. preprocess
transformed_sms = transform_text(input_sms)
# 2. vectorize
vector_input = tfidf.transform([transformed_sms])
# 3. predict
result = model.predict(vector_input)[0]
# 4. Display
if result == 1:
    st.header("Spam")
else:
    st.header("Not Spam")

