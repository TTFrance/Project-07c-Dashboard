"""
# My first app
Here's our first attempt at using data to create a table:
"""

# cd C:\Users\adam_\Desktop\OC\Project 07\Project-07c-Dashboard
# streamlit run firstslapp.py

import streamlit as st
import os

if "OCP7_API_URL" in os.environ:
    base_api_url = os.environ["OCP7_API_URL"]
    'OCP7_API_URL', base_api_url
else:
    'NOT FOUND'


tab1, tab2, tab3 = st.tabs(["EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_SINCE"])
sb = st.sidebar

with tab1:
   st.header("EXT_SOURCE_2")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
   st.header("EXT_SOURCE_3")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

with sb:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )