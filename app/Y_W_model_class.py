import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import gzip, pickle, pickletools

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import string
from bs4 import BeautifulSoup
import regex as re
from nltk.corpus import stopwords

def y_w_model(reddit_submission):
    #open RF model
    with gzip.open('code/Y_W_model.pkl', 'rb') as file:
        p = pickle.Unpickler(file)
        y_w_model = p.load()

    #open vectorizer
    cv = pickle.load(open('code/vector.pickle', 'rb'))

    if reddit_submission:
        X = reddit_submission
        cleaned_reddit_submission = remove_HTML_punc_lower_stop_l(X)
        X1 = [cleaned_reddit_submission]
        X1_cv = cv.transform(X1)
        y_pred = y_w_model.predict(X1_cv)

    return y_pred[0]

def remove_HTML_punc_lower_stop_l(text):
    #function to convert a title or self text into a single string

    #1.Remove HTML
    rtext = BeautifulSoup(text, features="lxml").get_text()

    #2. Remove non-letters
    letters_only = re.sub('[^a-zA-Z]',' ', rtext)

    #3. Convert to lower case, splits into individual words
    words = letters_only.lower().split()

    #4. Remove Stopwords
    stops = set(stopwords.words('english'))

    meaningful_words = [wordnet_lemmatizer.lemmatize(w) for w in words if w not in stops]

    return (' '.join(meaningful_words))
