"""This is the from which the web UI is created and displayed."""

import streamlit as st

st.set_page_config(
    layout='wide',
    page_icon='👩‍🏫 ',
    page_title='ATI'
)

st.title("Welcome to Ati! 📑")
st.subheader("A system for multiclass single-label Authorship aTtrIbution")

text_input = st.text_area('Text to analyze', height=450)

if st.button("Submit") and text_input != '':
    st.write("You entered:", text_input)
