"""This is the from which the web UI is created and displayed."""

import pickle
import pandas as pd
import streamlit as st

from pycaret.classification import load_model, predict_model
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing import tokenize

DATA_PATH_PREP = '../DATA/prepared'


def identity(x):
    return x


filename = f'{DATA_PATH_PREP}/04_vectorizer_hard.pkl'
with open(filename, 'rb') as f:
    vectorizer = pickle.load(f)


# class Tokenizer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X_cp = X.copy()
#         X_cp['text'] = X_cp['text'].map(lambda text: tokenize(text, 'hard'))
#         return X_cp


# class Vectorizer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         return vectorizer.transform(X['text'])


# filename = f'{DATA_PATH_PREP}/06_pipe_hard.pkl'
# with open(filename, 'rb') as f:
#     pipe_svc = pickle.load(f)

ridge = load_model(f'{DATA_PATH_PREP}/06_pycaret_ridge')
# et = load_model(f'{DATA_PATH_PREP}/06_pycaret_ridge_textfeats_et')

st.set_page_config(
    layout='wide',
    page_icon='👩‍🏫',
    page_title='ATI'
)

st.title("Welcome to Ati! 📑")
st.subheader("A system for multiclass single-label Authorship aTtrIbution")

text_input = st.text_area('Text to analyze', height=450)

if st.button("Submit") and text_input != '':

    with st.spinner():
        # Tokenize
        tokens = tokenize(text_input, 'hard')

        # Embed
        text_input_df_vect = pd.DataFrame(vectorizer.transform(
            [tokens]).toarray(), columns=vectorizer.get_feature_names())

        # Predict
        results_ridge = predict_model(ridge, data=text_input_df_vect, )
        prediction_ridge = results_ridge['prediction_label'][0]

        # results_et = predict_model(et, data=text_input_df_vect, )
        # print(results_et.columns)
        # prediction_et = results_et['prediction_label'][0]

    st.markdown(
        f'The most likely author according to ridge of the above text is **{prediction_ridge}**.')

    # st.markdown(
    #     f'The most likely author according to et of the above text is **{prediction_et}**.')

    # y_pred_class = model.predict(X_test) # Used in evalution
    # pipe_svc.predict(tmp)
    # print(pipe_svc.predict(pd.DataFrame({'text': text_input})))
    # preprcessed = pipeline(text_input)
    # prediction = predict_model(ridge, data=preprcessed)
