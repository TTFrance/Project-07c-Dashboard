"""
# My first app
Here's our first attempt at using data to create a table:
"""

# cd C:\Users\adam_\Desktop\OC\Project 07\Project-07c-Dashboard
# streamlit run firstslapp.py

import streamlit as st
import os

genre = st.radio("Low Risk",('111', '222', '333'),key='radio1')
genre = st.radio("Med Risk",('444', '555', '666'),key='radio2')
genre = st.radio("High Risk",('777', '888', '999'),key='radio3')

st.button('Say hello',key='btn')

