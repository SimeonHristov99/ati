"""This is the from which the web UI is created and displayed."""

import pickle
import pandas as pd
import streamlit as st

from pycaret.classification import load_model
from sklearn.base import BaseEstimator, TransformerMixin

class Tokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_cp = X.copy()
        X_cp['text'] = X_cp['text'].map(lambda text: tokenize(text, 'hard'))
        return X_cp

    
class Vectorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return vectorizer.transform(X['text'])


DATA_PATH_PREP = '../DATA/prepared'

pipe_svc = pickle.load(open(f'{DATA_PATH_PREP}/06_pipe_hard.pkl', 'rb'))
ridge = load_model(f'{DATA_PATH_PREP}/06_pycaret_ridge')

st.set_page_config(
    layout='wide',
    page_icon='👩‍🏫',
    page_title='ATI'
)

st.title("Welcome to Ati! 📑")
st.subheader("A system for multiclass single-label Authorship aTtrIbution")

text_input = st.text_area('Text to analyze', height=450)

if st.button("Submit") and text_input != '':
    st.write("You entered:", text_input)
    print(pd.DataFrame({'text': text_input}))
    # print(pipe_svc.predict(pd.DataFrame({'text': text_input})))
    # preprcessed = pipeline(text_input)
    # results = predict_model(ridge, data=preprcessed)
