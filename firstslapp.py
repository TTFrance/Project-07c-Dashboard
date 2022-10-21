"""
# My first app
Here's our first attempt at using data to create a table:
"""

# cd C:\Users\adam_\Desktop\OC\Project 07\Project-07c-Dashboard
# streamlit run testapp.py

import streamlit as st
import numpy as np
import requests
import json

# CONSTANTS
# url to make prediction
predict_url_slug = "https://ttfrance-firstapp.herokuapp.com/predict/"

# INITIALISATION
# init the client ids available
response = requests.get("https://ttfrance-firstapp.herokuapp.com/getids/")
lst_ids = []
for x in response.json(): 
    lst_ids.append(str(x))
arr_ids = np.array(lst_ids)

left_column, right_column = st.columns(2)
sidebar = st.sidebar

with sidebar:
    st.title('Client Loan Default')
    client_id = st.sidebar.selectbox('Client ID',arr_ids)
    'Selected ID: ', client_id

predict_url_full = predict_url_slug + client_id
response = requests.get(predict_url_full)
json_data = json.loads(response.text)
default_proba = round(json_data['prediction'],3)
default_proba_str = str(round(default_proba, 3)) + ' %'

# LEFT COLUMN
with left_column:
    'Default Probability', default_proba, "%"
    
# RIGHT COLUMN
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
