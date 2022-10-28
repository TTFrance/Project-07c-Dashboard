"""
# My first app
Here's our first attempt at using data to create a table:
"""

# cd C:\Users\adam_\Desktop\OC\Project 07\Project-07c-Dashboard
# streamlit run firstslapp.py

import streamlit as st
import pandas as pd


df = pd.read_pickle('data/client_data_api_dashboard_1k.pkl')

client_id = 226060

EXT_SOURCE_2 = df.loc[[int(client_id)]]['EXT_SOURCE_2'].values[0]
EXT_SOURCE_3 = df.loc[[int(client_id)]]['EXT_SOURCE_3'].values[0]
PAYMENT_RATE = df.loc[[int(client_id)]]['PAYMENT_RATE'].values[0]
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

CODE_GENDER_F='Female' if (df.loc[[int(client_id)]]['CODE_GENDER_F'].values[0]==1) else 'Male'
HigherEducation='Yes' if df.loc[[int(client_id)]]['NAME_EDUCATION_TYPE_Highereducation'].values[0]==1 else 'No'
Married='Yes' if df.loc[[int(client_id)]]['NAME_FAMILY_STATUS_Married'].values[0]==1 else 'No'

df_display = pd.DataFrame(columns=['value'])
df_display.loc['client_id'] = str(client_id)
df_display.loc['Gender'] = CODE_GENDER_F
df_display.loc['EXT_SOURCE_2'] = str(round(EXT_SOURCE_2,4))
df_display.loc['EXT_SOURCE_3'] = str(round(EXT_SOURCE_3,4))
df_display.loc['HigherEducation'] = HigherEducation
df_display.loc['Married'] = Married
df_display
