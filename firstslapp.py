"""
# My first app
Here's our first attempt at using data to create a table:
"""

# cd C:\Users\adam_\Desktop\OC\Project 07\Project-07c-Dashboard
# streamlit run firstslapp.py

import numpy as np
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

import streamlit as st
st.set_page_config(layout="wide")

# GRAPH SETTINGS
plt.style.use("dark_background")

# CONSTANTS
# base API url
# base_api_url = "https://ttfrance-firstapp.herokuapp.com/" # live
# base_api_url = "http://127.0.0.1:8000/" # local
base_api_url = os.environ['OCP7_API_URL']

# load the model
model = joblib.load('model/best_LGB_10k_Undersampled_BestParams.pkl')

# url to make prediction
predict_url_slug = base_api_url + "predict/"

# INITIALISATION
# init the client ids available
# response = requests.get(base_api_url + "getids/")

# lst_ids = []
# for x in response.json(): 
#     lst_ids.append(str(x))
# arr_ids = np.array(lst_ids)

# DATA FOR GRAPHING
df = pd.read_pickle('data/client_data_api_dashboard_1k.pkl')

# TEST
dictionary = {'A: Low': ['226060', '310013'], 'B: Medium': ['324973', '449031'], 'C: High': ['328692', '392503']}


sidebar = st.sidebar
tab1, tab2, tab3 = st.tabs(["Client Basic Info", "Main Feature Distributions", "Extra"])

with sidebar:
    st.title('Client Loan Default')
    #client_id = st.sidebar.selectbox('Client ID',arr_ids)
    #client_id = st.sidebar.text_input('Client ID',value="226060")
    # 'Selected ID: ', client_id

    # FIELDS TO SELECT CLIENT ID
    selected_section = st.selectbox("Risk Type:", sorted(dictionary.keys()))
    client_id = st.radio("Client:", sorted(dictionary[selected_section]))

    # GET PREDICTION PROBA %
    predict_url_full = predict_url_slug + client_id
    response = requests.get(predict_url_full)
    json_data = json.loads(response.text)
    default_proba = round(json_data['prediction'],3)
    default_proba_str = str(round(default_proba, 3)) + ' %'

    # SHOW API USED
    'USING API', base_api_url
    
    # END SIDEBAR

# EXTRACT SPECIFIC CLIENT KEY DATA VALUES
EXT_SOURCE_2 = df.loc[[int(client_id)]]['EXT_SOURCE_2'].values[0]
EXT_SOURCE_3 = df.loc[[int(client_id)]]['EXT_SOURCE_3'].values[0]
PAYMENT_RATE = df.loc[[int(client_id)]]['PAYMENT_RATE'].values[0]
CODE_GENDER_F = df.loc[[int(client_id)]]['CODE_GENDER_F'].values[0]
PA_PrLI_DELAY_DAYS_INSTALMENT__max__max = df.loc[[int(client_id)]]['PA_PrLI_DELAY_DAYS_INSTALMENT__max__max'].values[0]

DAYS_BIRTH = df.loc[[int(client_id)]]['DAYS_BIRTH'].values[0]
DAYS_EMPLOYED_PERC = df.loc[[int(client_id)]]['DAYS_EMPLOYED_PERC'].values[0]

AMT_CREDIT = df.loc[[int(client_id)]]['AMT_CREDIT'].values[0]

PA_DAYS_PROLONG_PCT__mean = df.loc[[int(client_id)]]['PA_DAYS_PROLONG_PCT__mean'].values[0]
PA_DAYS_TOT_DURATION__mean = df.loc[[int(client_id)]]['PA_DAYS_TOT_DURATION__mean'].values[0]
PA_DAYS_FIRST_DUE__min = df.loc[[int(client_id)]]['PA_DAYS_FIRST_DUE__min'].values[0]
Bur_CB_AMT_CREDIT_SUM_DEBT__sum = df.loc[[int(client_id)]]['Bur_CB_AMT_CREDIT_SUM_DEBT__sum'].values[0]
PA_AMT_DIFF_PCT__mean = df.loc[[int(client_id)]]['PA_AMT_DIFF_PCT__mean'].values[0]
Bur_CB_DAYS_CREDIT_ENDDATE__max = df.loc[[int(client_id)]]['Bur_CB_DAYS_CREDIT_ENDDATE__max'].values[0]
NAME_EDUCATION_TYPE_Highereducation = df.loc[[int(client_id)]]['NAME_EDUCATION_TYPE_Highereducation'].values[0]
NAME_FAMILY_STATUS_Married = df.loc[[int(client_id)]]['NAME_FAMILY_STATUS_Married'].values[0]

with tab1:
    tab1_left_column, tab1_right_column = st.columns([1, 2], gap="small")
    with tab1_left_column:
        st.subheader('Basic Details')
        'Default Probability', default_proba, "%"
        'AGE:', str(round(abs(DAYS_BIRTH)/365.25,0)),'years'
        'DAYS_EMPLOYED_PERC', str(round(DAYS_EMPLOYED_PERC,2)),'%'
        if CODE_GENDER_F == 1:
            "SEX:", "Female"
        else:
            "SEX:","Male"
        if NAME_EDUCATION_TYPE_Highereducation == 1:
            "Higher Education:", "Yes"
        else:
            "Higher Education:", "No"
        if NAME_FAMILY_STATUS_Married == 1:
            "Marital Status:", "Married"
        else:
            "Marital Status:", "Not Married"
    # END TAB1 LEFT COLUMN

    with tab1_right_column:
        st.subheader('Feature Importances')
        features = pd.DataFrame(columns=['name','importance'])
        importance = model.feature_importances_
        for i,v in enumerate(importance):
            feature_name = model.feature_name_[i]
            features.loc[len(features.index)] = [feature_name,v] 
        features = features.sort_values('importance', ascending=False).head(15) #.set_index('name')

        fig, ax = plt.subplots()
        ax.barh(data=features.sort_values('importance'),y='name',width='importance')
        fig

    # END TAB1 RIGHT COLUMN

with tab2:
    tab2_left_column, tab2_right_column = st.columns([1, 1], gap="small")
    with tab2_left_column:

        # Group data together
        #EXT_SOURCE_2_data = df.EXT_SOURCE_2.values
        #EXT_SOURCE_2_data2 = []
        #EXT_SOURCE_2_data2.append(EXT_SOURCE_2_data)
        #group_labels = ['EXT_SOURCE_2']
        # Create distplot with custom bin_size
        #fig = ff.create_distplot(EXT_SOURCE_2_data2, group_labels, bin_size=[.05])
        # Plot!
        #st.plotly_chart(fig, use_container_width=True)

        'EXT_SOURCE_2', round(EXT_SOURCE_2,4)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="EXT_SOURCE_2", fill=True, color='blue').set(title='Dist EXT_SOURCE_2')
        plt.axvline(x=EXT_SOURCE_2, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)

        'EXT_SOURCE_3', round(EXT_SOURCE_3,4)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="EXT_SOURCE_3", fill=True, color='orange').set(title='Dist EXT_SOURCE_3')
        plt.axvline(x=EXT_SOURCE_3, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)
    # END TAB2 LEFT COLUMN

    # RIGHT COLUMN
    with tab2_right_column:

        'PAYMENT_RATE', round(PAYMENT_RATE,4)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="PAYMENT_RATE", fill=True, color='green').set(title='Dist PAYMENT_RATE')
        plt.axvline(x=PAYMENT_RATE, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)

        'AMT_CREDIT', round(AMT_CREDIT,4)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="AMT_CREDIT", fill=True, color='yellow').set(title='Dist AMT_CREDIT')
        plt.axvline(x=AMT_CREDIT, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)

    # END TAB2 RIGHT COLUMN

with tab3:
    tab3_left_column, tab3_right_column = st.columns([1, 1], gap="small")
    with tab3_left_column:

        
        #'Max PA_PrLI_DELAY_DAYS_INSTALMENT', round(PA_PrLI_DELAY_DAYS_INSTALMENT__max__max,4)
        #fig = plt.figure(figsize=(6, 4))
        #sns.kdeplot(data=df, x="PA_PrLI_DELAY_DAYS_INSTALMENT__max__max", fill=True, color='yellow').set(title='Dist Max PA_PrLI_DELAY_DAYS_INSTALMENT')
        #plt.axvline(x=PAYMENT_RATE, color='red', linestyle='--', linewidth=2, alpha=0.5)
        #st.pyplot(fig)

        'GRAPH 6 PLACEHOLDER','Something'
    # END TAB3 LEFT COLUMN
    with tab3_right_column:
        'GRAPH 7 PLACEHOLDER','Something'
        'GRAPH 8 PLACEHOLDER','Something'
    # END TAB3 RIGHT COLUMN
    
