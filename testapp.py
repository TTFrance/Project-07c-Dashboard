"""
# My first app
Here's our first attempt at using data to create a table:
"""

# cd C:\Users\adam_\Desktop\OC\Project 07\Project-07c-Dashboard
# streamlit run firstslapp.py

import streamlit as st
import numpy as np
import pandas as pd
import requests

# FUNCTIONS
def get_model_prediction():
    predict_url_full = predict_url_slug + st.session_state.client_id
    response = requests.get(predict_url_full)
    st.write(response.json())
    st.session_state.client_id = client_id

# CONSTANTS
# url to make prediction
predict_url_slug = "https://ttfrance-firstapp.herokuapp.com/predict/"


# INITIALISATION
# init the client ids available
response = requests.get("https://ttfrance-firstapp.herokuapp.com/getids/123")
lst_ids = []
for x in response.json(): 
    lst_ids.append(str(x))
arr_ids = np.array(lst_ids)


# SESSION STATE
if 'client_id' not in st.session_state:
    st.write("WE HAVE NO CLIENT ID")
    st.session_state.client_id = ''
else:
    st.write("WE HAVE CLIENT ID")
    st.write(st.session_state.client_id)
    get_model_prediction()



# LAYOUT
left_column, right_column = st.columns(2)
sidebar = st.sidebar

with sidebar:
    client_id = st.sidebar.selectbox(
        'Client ID',
        lst_ids,
        on_change=get_model_prediction
        )
    'Selected ID: ', client_id

# LEFT COLUMN
with left_column:
    'Just a placeholder', 999
    
# RIGHT COLUMN
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")


#chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])
#st.line_chart(chart_data)

#map_data = pd.DataFrame(
#    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#    columns=['lat', 'lon'])
#st.map(map_data)

#x = st.slider('x')  # 👈 this is a widget
#st.write(x, 'squared is', x * x)

#st.text_input("Your name", key="name")
#st.session_state.name


