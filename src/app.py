"""This is the from which the web UI is created and displayed."""

import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

from pipelines import pipeline_tf_idf

DATA_PATH_PREP = '../DATA/prepared'


def identity(x):
    return x


ridge = load_model(f'{DATA_PATH_PREP}/06_pycaret_ridge')
# et = load_model(f'{DATA_PATH_PREP}/06_pycaret_ridge_textfeats_et')

st.set_page_config(
    layout='wide',
    page_icon='random',
    page_title='Ati'
)


st.title("Welcome to Ati! 📑")
st.subheader("A system for multiclass single-label Authorship aTtrIbution")

text_input = st.text_area('Text to analyze', height=450)

if st.button("Submit") and text_input != '':

    with st.spinner():
        text_input_df_vect = pipeline_tf_idf(text_input)
        # print(text_input_df_vect)

        # Predict
        results_ridge = predict_model(ridge, data=text_input_df_vect)
        prediction_ridge = results_ridge['prediction_label'][0]

        # results_et = predict_model(et, data=text_input_df_vect, )
        # print(results_et.columns)
        # prediction_et = results_et['prediction_label'][0]

    # st.write('Done')
    st.markdown(
        f'The most likely author according to ridge of the above text is **{prediction_ridge}**.')

    # st.markdown(
    #     f'The most likely author according to et of the above text is **{prediction_et}**.')

    # y_pred_class = model.predict(X_test) # Used in evalution
    # pipe_svc.predict(tmp)
    # print(pipe_svc.predict(pd.DataFrame({'text': text_input})))
    # preprcessed = pipeline(text_input)
    # prediction = predict_model(ridge, data=preprcessed)
