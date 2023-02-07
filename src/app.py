"""This is the from which the web UI is created and displayed."""

import numpy as np
import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

from pipelines import pipeline_sbert, pipeline_text_features, pipeline_tf_idf

DATA_PATH_PREP = '../DATA/prepared'


def identity(x):
    return x


ridge = load_model(f'{DATA_PATH_PREP}/06_pycaret_ridge')
et = load_model(f'{DATA_PATH_PREP}/06_pycaret_ridge_textfeats_et')
ridge_sbert = load_model(f'{DATA_PATH_PREP}/06_pycaret_sbert')

st.set_page_config(
    layout='wide',
    page_icon='random',
    page_title='Ati'
)

st.title("Welcome to Ati! 📑")
st.subheader("A system for multiclass single-label Authorship aTtrIbution")

with st.sidebar:
    add_radio = st.radio(
        'Choose a text preprocessing method',
        ('TF-IDF', 'Text Features', 'sBERT')
    )

text_input = st.text_area('Text to analyze', height=450)

if st.button("Submit") and text_input != '':

    with st.spinner():
        if add_radio == 'TF-IDF':
            text_input_df_vect = pipeline_tf_idf(text_input)
            results_ridge = predict_model(ridge, data=text_input_df_vect)
            prediction_ridge = results_ridge['prediction_label'][0]
            st.markdown(
                f'The most likely author according to Ridge Classifier of the above text is **{prediction_ridge}**.')
        elif add_radio == 'Text Features':
            text_input_tf, tags = pipeline_text_features(
                text_input, return_tags=True)

            st.write(
                'Here is a breakdown of the part-of-speech tags found in your text:')
            st.write(tags)

            st.write(
                'Here is the complexity of the text based on different metrics:')
            metrics = ['fre', 'air', 'gfi', 'cli', 'smog']
            st.write(text_input_tf[metrics])

            text_input_tf.columns = np.arange(len(text_input_tf.columns))
            results_et = predict_model(et, data=text_input_tf)
            prediction_et = results_et['prediction_label'][0]
            st.markdown(
                f'The most likely author according to Extra Trees Classifier of the above text is **{prediction_et}**.')
        elif add_radio == 'sBERT':
            text_input_sbert = pipeline_sbert(text_input)
            print(text_input_sbert)
            results_sbert = predict_model(ridge_sbert, data=text_input_sbert)
            prediction_sbert = results_sbert['prediction_label'][0]
            st.markdown(
                f'The most likely author according to Ridge Classifier of the above text is **{prediction_sbert}**.')
        else:
            st.error('This type of text preprocessing is not supported, yet.')

    # y_pred_class = model.predict(X_test) # Used in evalution
    # pipe_svc.predict(tmp)
    # print(pipe_svc.predict(pd.DataFrame({'text': text_input})))
    # preprcessed = pipeline(text_input)
    # prediction = predict_model(ridge, data=preprcessed)
