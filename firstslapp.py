"""
# My first app
Here's our first attempt at using data to create a table:
"""

# cd C:\Users\adam_\Desktop\OC\Project 07\Project-07c-Dashboard
# streamlit run firstslapp.py

import joblib
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import seaborn as sb
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
import shap
import streamlit as st
from streamlit_shap import st_shap
import sys
# import streamlit.components.v1 as components

# STREAMLIT WIDTH CONFIG
st.set_page_config(layout="wide")

# PATH TO BE ABLE TO PICK UP ENV VARIABLE 'OCP7_API_URL'
sys.path.append(".")

# GRAPH SETTINGS
plt.style.use('Solarize_Light2')
plt.style.use("dark_background")
plt.style.use('default')

# URL FOR API, FROM ENV VARIABLE 'OCP7_API_URL'
# base_api_url = "https://ttfrance-firstapp.herokuapp.com/" # live
# base_api_url = "http://127.0.0.1:8000/" # local
base_api_url = os.environ['OCP7_API_URL']

# LOAD THE PREDICTION MODEL (NOT USED HERE FOR A SINGLE PREDICTION)
model = joblib.load('model/best_LGB_10k_Undersampled_BestParams.pkl')

# URI FOR PREDICTION API
predict_url_slug = base_api_url + "predict/"

# DATA FOR GRAPHING
df = pd.read_pickle('data/client_data_api_dashboard_1k.pkl')

# LOAD CURVES FOR PROFIT / LOSS
CURVES = pd.read_pickle('data/curves.pkl')
CURVES2 = pd.read_pickle('data/curves_new.pkl')

# ANNUITY AMOUNTS
df_amt_annuity = pd.read_pickle('data/df_amt_annuity.pkl')

# TEST
dictionary = {'A: Low': ['226060', '310013'], 'B: Medium': ['324973', '449031'], 'C: High': ['328692', '392503']}

# CREATE THE SIDEBAR
sidebar = st.sidebar

# CREATE MAIN TABS
tab1, tab2, tab3 = st.tabs(["Explanations", "Feature Scatterplots", "Main Features"])

# functions
def plot_shap_summary_feature(shap_values,X,feature_name):
    X_array = X.columns.to_numpy()
    pos = np.where(X_array == feature_name)[0][0]
    plt.clf()
    fig = shap.summary_plot(shap_values[:,pos:pos+1], X.iloc[:, pos:pos+1], plot_size=[6.0,4.0], plot_type='violin')
    st.pyplot(fig, matplotlib=True)

def get_feature_index(feature_name,X):
    X_array = X.columns.to_numpy()
    return np.where(X_array == feature_name)[0][0]

# SIDEBAR
with sidebar:
    st.title('Client Loan Default')
    #client_id = st.sidebar.selectbox('Client ID',arr_ids)
    #client_id = st.sidebar.text_input('Client ID',value="226060")
    # 'Selected ID: ', client_id

    # FIELDS TO SELECT CLIENT ID
    #selected_section = st.selectbox("Risk Type:", sorted(dictionary.keys()))
    #client_id = st.radio("Client:", sorted(dictionary[selected_section]))

    client_id = st.selectbox(
        'Client ID',
        ('226060', '310013', '324973', '449031', '328692', '392503')
    )
    # st.write('Chosen:', client_id)

    # GET PREDICTION PROBA %
    predict_url_full = predict_url_slug + client_id
    response = requests.get(predict_url_full)
    json_data = json.loads(response.text)
    default_proba = round(json_data['prediction'],3)
    default_proba_str = str(round(default_proba, 3)) + ' %'

    # EXTRACT SPECIFIC CLIENT KEY DATA VALUES
    EXT_SOURCE_2 = df.loc[[int(client_id)]]['EXT_SOURCE_2'].values[0]
    EXT_SOURCE_3 = df.loc[[int(client_id)]]['EXT_SOURCE_3'].values[0]
    PAYMENT_RATE = df.loc[[int(client_id)]]['PAYMENT_RATE'].values[0]
    CODE_GENDER_F = df.loc[[int(client_id)]]['CODE_GENDER_F'].values[0]
    PA_PrLI_DELAY_DAYS_INSTALMENT__max__max = df.loc[[int(client_id)]]['PA_PrLI_DELAY_DAYS_INSTALMENT__max__max'].values[0]
    DAYS_BIRTH = df.loc[[int(client_id)]]['DAYS_BIRTH'].values[0]
    DAYS_EMPLOYED_PERC = df.loc[[int(client_id)]]['DAYS_EMPLOYED_PERC'].values[0]
    AMT_CREDIT = df.loc[[int(client_id)]]['AMT_CREDIT'].values[0]
    AMT_ANNUITY = df_amt_annuity[df_amt_annuity.index == int(client_id)]['AMT_ANNUITY'].values[0]
    PA_DAYS_PROLONG_PCT__mean = df.loc[[int(client_id)]]['PA_DAYS_PROLONG_PCT__mean'].values[0]
    PA_DAYS_TOT_DURATION__mean = df.loc[[int(client_id)]]['PA_DAYS_TOT_DURATION__mean'].values[0]
    PA_DAYS_FIRST_DUE__min = df.loc[[int(client_id)]]['PA_DAYS_FIRST_DUE__min'].values[0]
    Bur_CB_AMT_CREDIT_SUM_DEBT__sum = df.loc[[int(client_id)]]['Bur_CB_AMT_CREDIT_SUM_DEBT__sum'].values[0]
    PA_AMT_DIFF_PCT__mean = df.loc[[int(client_id)]]['PA_AMT_DIFF_PCT__mean'].values[0]
    Bur_CB_DAYS_CREDIT_ENDDATE__max = df.loc[[int(client_id)]]['Bur_CB_DAYS_CREDIT_ENDDATE__max'].values[0]
    NAME_EDUCATION_TYPE_Highereducation = df.loc[[int(client_id)]]['NAME_EDUCATION_TYPE_Highereducation'].values[0]
    NAME_FAMILY_STATUS_Married = df.loc[[int(client_id)]]['NAME_FAMILY_STATUS_Married'].values[0]

    # CALCULATE LOSS/GAIN CURVE
    W0 = AMT_ANNUITY
    W1 = AMT_CREDIT
    CURVES['EARNED']     = W0*CURVES.TN
    CURVES['NOT_EARNED'] = W0*CURVES.FP
    CURVES['LOST']       = W1*CURVES.FN
    CURVES['GAIN']       = CURVES.EARNED - CURVES.LOST
    CURVES['MAX_GAIN']   = CURVES.EARNED + CURVES.NOT_EARNED

    # SIDEBAR SHOW BASIC CLIENT DETAILS
    st.subheader('Basic Details')
    st.markdown('Default Probability: **'+ str(default_proba*100) + '** %')
    st.markdown('Age: **'+ str(round(abs(DAYS_BIRTH)/365.25,0)) + '** Years')
    st.markdown('% Days Employed: **'+ str(round(DAYS_EMPLOYED_PERC,2)) + '**')
    if CODE_GENDER_F == 1:
        st.markdown('Sex: **Female**')
    else:
        st.markdown('Sex: **Male**')
    if NAME_EDUCATION_TYPE_Highereducation == 1:
        st.markdown('Higher Education: **Yes**')
    else:
        st.markdown('Higher Education: **No**')
    if NAME_FAMILY_STATUS_Married == 1:
        st.markdown('Marital Status: **Married**')
    else:
        st.markdown('Marital Status: **Not Married**')

    # SHOW API USED
    'USING API', base_api_url+'docs'

    # END SIDEBAR


# TAB 1 - EXPLANATIONS
with tab1:
    #features = pd.DataFrame(columns=['name','importance'])
    #importance = model.feature_importances_
    #for i,v in enumerate(importance):
    #    feature_name = model.feature_name_[i]
    #    features.loc[len(features.index)] = [feature_name,v] 
    #features = features.sort_values('importance', ascending=False).head(15) #.set_index('name')
    #fig, ax = plt.subplots()
    #ax.barh(data=features.sort_values('importance'),y='name',width='importance')
    #st.pyplot(fig, matplotlib=True)

    tab1_1, tab1_2, tab1_3 = st.tabs(["Local Explanation", "Global Feature Importance", "Risk Thresholds"])

    # SHAP WATERFALL
    with tab1_1:
        st.subheader('Local Explanation')
        X = df.drop(columns=['TARGET'])
        df2 = df.index.to_frame().reset_index(drop=True)
        pos_num = df2[df2['SK_ID_CURR'] == int(client_id)].index[0]
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        fig = shap.plots.waterfall(shap_values[pos_num], max_display=9)
        st.pyplot(fig, matplotlib=True)

        st.success('**The Local Explanation:** \
            \n\nThis shows the effect that each feature has on a single prediction in the model. \
            \n\nThe bottom of a waterfall plot starts as the expected value of the model output,  \
            and then each row shows how the positive (red) or negative (blue) contribution of each  \
            feature moves the value from the expected model output over the background dataset \
            to the model output for this prediction. \
            \n\nA blue arrow pointing to the left moves the prediction towards being less likely of default, \
            and a red arrow pointing to the right moves the prediction towards being more likely')


    with tab1_2:
        st.subheader('Global Feature Importance')
        # -----------------
        # SHAP
        # DO NOT TOUCH THIS
        # DO NOT TOUCH THIS
        # DO NOT TOUCH THIS
        X = df.drop(columns=['TARGET'])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # LOCAL EXPLANATION
        # STREAMLIT_SHAP
        #st.subheader('Local Explanation')
        #st_shap.st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X.iloc[0,:]))
        # GLOBAL EXPLANATION
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #fig = shap.summary_plot(shap_values[1], X, show = False, cmap=plt.get_cmap("viridis_r"))
        fig = shap.summary_plot(shap_values[1], X, show = False, plot_type='violin')
        st.pyplot(fig, matplotlib=True)
        # DO NOT TOUCH THIS
        # DO NOT TOUCH THIS
        # DO NOT TOUCH THIS
        # -----------------
        st.success('**To interpret a SHAP model explainer**: \n\n Higher feature values are in red, \
            lower are in blue. Taking EXT_SOURCE_2 as an example we can see that higher values \
            result in a bigger negative effect on the model prediction (probability of default). \
            \n\n Therefore clients with higher EXT_SOURCE_2 values are less likely to default.')

    with tab1_3:
        st.subheader('Risk Thresholds')
        # GAIN / LOSS CURVES GRAPH
        #st.subheader('Gain / Loss Curves')
        #fig, ax = plt.subplots(figsize =(10, 6))
        #CURVES.plot(ax=ax, x='threshold', y=['EARNED', 'LOST'])
        #plt.axvline(x=default_proba, color='red', linestyle='--', linewidth=2, alpha=0.5)
        #st.pyplot(fig)

        # CURVES NEW
        risk = st.selectbox(
            'Select your acceptable level of risk:',
            ('Low', 'Medium', 'High')
        )

        risk_medium_lower = 0.360
        risk_medium_upper = 0.565
        risk_band = 0.2

        risk_graph_title = 'Client placement within risk thresholds'

        if risk == 'Medium':
            risk_lower = risk_medium_lower
            risk_upper = risk_medium_upper
        elif risk == 'Low':
            risk_lower = risk_medium_lower - risk_band
            risk_upper = risk_medium_upper - risk_band
        else: # risk == high:
            risk_lower = risk_medium_lower + risk_band
            risk_upper = risk_medium_upper + risk_band

        COLORS = dict({'CONSTANT_COST':'blue', 'ACCEPTING_COST':'red', 'REJECTING_COST':'orange'})
        COMMON_GRAPH_OPTIONS = dict(x='threshold',color=COLORS)
        fig, ax = plt.subplots(figsize =(10, 6))
        CURVES2.plot.area(
            ax=ax,
            y=['CONSTANT_COST', 'REJECTING_COST', 'ACCEPTING_COST', ],
            stacked=True,
            **COMMON_GRAPH_OPTIONS
        )
        W0 =  27108  # penalty=12.5
        W1 =  599026  #
        ax.plot([0, 1],[-W0, -W0], ':')
        ax.set_xlim(0,1)
        ax.legend(loc='lower left')

        plt.axvline(x=default_proba, ymin=0, ymax=1, color='black', linestyle='--', linewidth=1, alpha=0.4)
        plt.axvline(x=risk_lower, ymin=0, ymax=1, color='green', linestyle='--', linewidth=1, alpha=0.4)
        plt.axvline(x=risk_upper, ymin=0, ymax=1, color='darkred', linestyle='--', linewidth=1, alpha=0.4)

        ax.annotate('low risk threshold',
                    xy=(risk_lower, -40000), xycoords='data', color='green',
                    xytext=(-15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    horizontalalignment='right', verticalalignment='bottom')

        ax.annotate('high risk threshold',
                    xy=(risk_upper, -40000), xycoords='data', color='darkred',
                    xytext=(15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='darkred', shrink=0.05),
                    horizontalalignment='left', verticalalignment='bottom')

        ax.annotate('client probability of default',
                    xy=(default_proba, -30000), xycoords='data', color='black',
                    xytext=(-15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='right', verticalalignment='bottom')

        plt.suptitle(risk_graph_title)
        st.pyplot(fig)

        # in which category is the client regarding the currently selected risk?
        if default_proba <= risk_lower:
            risk_client = 'Low'
        elif default_proba >= risk_upper:
            risk_client = 'High'
        else:
            risk_client = 'Medium'

        base_text = 'The green line is low risk threshold. \
                    Above the red line is high risk threshold. \
                    Inbetween the green and red lines is medium risk band. \
                    \n\nThe black line is the clients calculated probability of default.\n\n'

        if risk_client == 'Medium':
            extra_text = 'This client falls into the band of **medium** risk and you should \
                study the clients values for the most important features as shown in \
                study the clients values for the most important features as shown in \
                the 2nd and 3rd tabs, to draw a manual conclusion on whether or not to agree the loan.'
        elif risk_client == 'Low':
            extra_text = 'This client falls into the band of **low** risk and could be \
                **accepted** with no further manual analysis, however please take a moment to \
                study the clients values for the most important features as shown in \
                the 2nd and 3rd tabs, to make sure you agree with this recommendation.'
        else: # risk == high:
            extra_text = 'This client falls into the band of **high** risk and could be \
                rejected with no further manual analysis, however please take a moment to \
                study the clients values for the most important features as shown in \
                the 2nd and 3rd tabs, to make sure you agree with this recommendation.'

        risk_explanation_text = base_text + extra_text

        st.success(risk_explanation_text)



# TAB 2 - 4 HISTOGRAMS TOP 4 FEATURES
with tab2:
    st.subheader('Feature Comparison - Actual Values')
    df_shap_values = pd.DataFrame(data=shap_values[1],columns=df.drop(columns=['TARGET']).columns)
    df_feature_importance = pd.DataFrame(columns=['feature','importance'])
    for col in df_shap_values.columns:
        importance = df_shap_values[col].abs().mean()
        df_feature_importance.loc[len(df_feature_importance)] = [col,importance]
    df_feature_importance = df_feature_importance.sort_values('importance',ascending=False)
    top_10_features = df_feature_importance.feature.head(10).to_list()
    tab2_left_column, tab2_right_column = st.columns([1, 1], gap="small")
    with tab2_left_column:
        featureX = st.selectbox('Select X Feature:', top_10_features)
    with tab2_right_column:
        featureY = st.selectbox('Select Y Feature:', top_10_features)

    if featureX == featureY:
        st.warning('**Warning -** Comparison features are the same, KDE will not be shown.')
    # scatter of the two features
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1,1,1)

    # add single point for the chosen client
    df2 = df.index.to_frame().reset_index(drop=True)
    pos_num = df2[df2['SK_ID_CURR'] == int(client_id)].index

    cm = df.TARGET.map({0: 'green', 1: 'blue'})
    ax.scatter(data=df, x=featureX, y=featureY, s=0.5, c=cm) # c='yellow'
    if featureX != featureY:
        sb.kdeplot(data=df, x=featureX, y=featureY)
    ax.scatter(data=df.iloc[pos_num], x=featureX, y=featureY, c='red', s=20, marker='o')
    ax.set_xlabel(featureX)
    ax.set_ylabel(featureY)
    custom = [
                Line2D([], [], marker='.', markersize=5, color='red', linestyle='None'),
                Line2D([], [], marker='.', markersize=5, color='blue', linestyle='None')
            ]
    fig.subplots_adjust(right=0.8)
    plt.title('Actual Feature Values, Client ' + client_id)
    ax.legend(custom, ['Client', 'Others'], loc='center left', fontsize=12, bbox_to_anchor=(0.8, 0.5), bbox_transform=fig.transFigure)
    st.pyplot(fig)

    # NEW LAYOUT
    st.subheader('Feature Comparison - SHAP Values')
    fig = plt.figure(figsize=(12, 8))
    top_left = fig.add_subplot(2,2,1)
    top_left.xaxis.set_label_position('top') 
    top_left.set_xlabel(featureX)    
    top_left.set_ylabel(featureY)    
    top_left.scatter(data=df_shap_values, x=featureX, y=featureY, s=1, marker='.')
    top_left.scatter(data=df_shap_values.iloc[pos_num], x=featureX, y=featureY, c='red', s=10, marker='D')
    if featureX != featureY:
        sb.kdeplot(ax=top_left,data=df_shap_values, x=featureX, y=featureY)

    top_right = fig.add_subplot(2,2,2)
    # top_right.hist(df_shap_values[featureY], orientation="horizontal", bins=50)
    sb.kdeplot(ax=top_right, data=df_shap_values, y=featureY)
    top_right.axhline(y=df_shap_values.iloc[pos_num][featureY].values[0]
        , color='red', linestyle='--', linewidth=1, alpha=0.5)

    bot_left = fig.add_subplot(2,2,3)
    sb.kdeplot(ax=bot_left, data=df_shap_values, x=featureX)
    bot_left.axvline(x=df_shap_values.iloc[pos_num][featureX].values[0]
        , color='red', linestyle='--', linewidth=1, alpha=0.5)
    # bot_left.hist(df_shap_values[featureX], bins=50)

    fig.suptitle('SHAP Feature Values, Client ' + client_id)
    st.pyplot(fig)

# TAB 3 - 4 HISTOGRAMS 4 FEATURES
with tab3:
    tab3_left_column, tab3_right_column = st.columns([1, 1], gap="small")
    with tab3_left_column:

        # Group data together
        #EXT_SOURCE_2_data = df.EXT_SOURCE_2.values
        #EXT_SOURCE_2_data2 = []
        #EXT_SOURCE_2_data2.append(EXT_SOURCE_2_data)
        #group_labels = ['EXT_SOURCE_2']
        # Create distplot with custom bin_size
        #fig = ff.create_distplot(EXT_SOURCE_2_data2, group_labels, bin_size=[.05])
        # Plot!
        #st.plotly_chart(fig, use_container_width=True)


        st.subheader('EXT_SOURCE_2')
        st.markdown('Value: <span style="color:red">**' + str(round(EXT_SOURCE_2,4)) + '**</span>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="EXT_SOURCE_2", fill=True, color='blue').set(title='Dist EXT_SOURCE_2')
        plt.axvline(x=EXT_SOURCE_2, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)
        #plot_shap_summary_feature(shap_values[1],X,'EXT_SOURCE_2')
        #pos = get_feature_index('EXT_SOURCE_2',X)
        #fig = shap.summary_plot(shap_values[1][:,pos:pos+1], X.iloc[:, pos:pos+1])
        #st.pyplot(fig, matplotlib=True)


        st.subheader('EXT_SOURCE_3')
        st.markdown('Value: <span style="color:red">**' + str(round(EXT_SOURCE_3,4)) + '**</span>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="EXT_SOURCE_3", fill=True, color='orange').set(title='Dist EXT_SOURCE_3')
        plt.axvline(x=EXT_SOURCE_3, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)
        #plot_shap_summary_feature(shap_values[1],X,'EXT_SOURCE_3')

    # END TAB2 LEFT COLUMN

    # RIGHT COLUMN
    with tab3_right_column:

        st.subheader('PAYMENT_RATE')
        st.markdown('Value: <span style="color:red">**' + str(round(PAYMENT_RATE,4)) + '**</span>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="PAYMENT_RATE", fill=True, color='green').set(title='Dist PAYMENT_RATE')
        plt.axvline(x=PAYMENT_RATE, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)
        #plot_shap_summary_feature(shap_values[1],X,'PAYMENT_RATE')

        st.subheader('AMT_CREDIT')
        st.markdown('Value: <span style="color:red">**' + str(round(AMT_CREDIT,4)) + '**</span>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="AMT_CREDIT", fill=True, color='yellow').set(title='Dist AMT_CREDIT')
        plt.axvline(x=AMT_CREDIT, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)
        #plot_shap_summary_feature(shap_values[1],X,'AMT_CREDIT')
    # END TAB2 RIGHT COLUMN


        
