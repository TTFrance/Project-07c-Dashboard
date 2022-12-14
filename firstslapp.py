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
st.set_page_config(layout="wide", page_title='Loan Credit Default Risk, Adam Phillips', page_icon = 'favicon.ico')

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
df = pd.read_pickle('data/client_data_api_dashboard_1k_500g_500b.pkl')

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
tab1, tab2, tab3 = st.tabs(["Explications", "Analyse bivariée", "Caractéristiques principales"])

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
    st.subheader('Principales valeurs caractéristiques')
    #st.markdown('Default Probability: **'+ str(default_proba*100) + '** %')
    #st.markdown('Age: **'+ str(round(abs(DAYS_BIRTH)/365.25,0)) + '** Years')
    #st.markdown('% Days Employed: **'+ str(round(DAYS_EMPLOYED_PERC,2)) + '**')
    #if CODE_GENDER_F == 1:
    #    st.markdown('Sex: **Female**')
    #else:
    #    st.markdown('Sex: **Male**')
    #if NAME_EDUCATION_TYPE_Highereducation == 1:
    #    st.markdown('Higher Education: **Yes**')
    #else:
    #    st.markdown('Higher Education: **No**')
    #if NAME_FAMILY_STATUS_Married == 1:
    #    st.markdown('Marital Status: **Married**')
    #else:
    #    st.markdown('Marital Status: **Not Married**')

    t = pd.DataFrame(data={
        'Feature': [
            'Probabilité de défaut (%)',
            'Âge (ans)',
            '% Jours travaillés',
            'Sexe',
            'L`\'enseignement supérieur',
            'État civil',
        ], 
        'Value': [
            str(default_proba*100) + ' %',
            str(round(abs(DAYS_BIRTH)/365.25,0)),
            str(round(DAYS_EMPLOYED_PERC,2)),
            'Femelle' if CODE_GENDER_F == 1 else 'Mâle',
            'Oui' if NAME_EDUCATION_TYPE_Highereducation == 1 else 'Non',
            'Marié(e)' if NAME_FAMILY_STATUS_Married == 1 else 'Pas marié(e)',
        ]})

    hide_table_row_index = """
            <style>
            table thead tr th { font-weight : 700; color: rgba(0, 0, 0, 1.0)}
            thead tr th:first-child {display:none}
            tbody th {display:none}
            table td { border-style : hidden!important; }
            table th { border-style : hidden!important }
            </style>
            """
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    st.table(t)

    # SHOW API USED
    'API:', base_api_url+'docs'

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

    tab1_1, tab1_2, tab1_3 = st.tabs(["Analyse individuelle", "Analyse globale", "Seuils de risque"])

    # SHAP WATERFALL
    with tab1_1:
        st.subheader('Explication locale')
        X = df.drop(columns=['TARGET'])
        df2 = df.index.to_frame().reset_index(drop=True)
        pos_num = df2[df2['SK_ID_CURR'] == int(client_id)].index[0]
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        fig = shap.plots.waterfall(shap_values[pos_num], max_display=9)
        st.pyplot(fig, matplotlib=True)

        st.success('**Explication locale:** \
            \n\nCela montre l\'effet que chaque caractéristique a sur une seule prédiction dans le modèle. \
            \n\nLe bas d\'un tracé en cascade commence par la valeur attendue de la sortie du modèle, \
            puis chaque ligne montre comment la contribution positive (rouge) ou négative (bleue) de  \
            chaque entité déplace la valeur de la sortie attendue du modèle sur l\'ensemble de données  \
            \'arrière-plan vers le sortie du modèle pour cette prédiction. \
            \n\nUne flèche bleue pointant vers la gauche déplace la prédiction vers une probabilité de défaut moindre, \
            et une flèche rouge pointant vers la droite déplace la prédiction vers une probabilité plus élevée')


    with tab1_2:
        st.subheader('Importance globale des caractéristiques')
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
        st.success('**Importance globale des caractéristiques**: \
            \n\nLes valeurs de fonctionnalité supérieures sont en rouge, les valeurs inférieures en bleu. \
            En prenant EXT_SOURCE_2 comme exemple, nous pouvons voir que des valeurs plus élevées entraînent \
            un effet négatif plus important sur la prédiction du modèle (probabilité de défaut). \
            \n\nPar conséquent, les clients avec des valeurs EXT_SOURCE_2 plus élevées sont moins susceptibles d\'être par défaut.')

    with tab1_3:
        st.subheader('Seuils de risque')
        # GAIN / LOSS CURVES GRAPH
        #st.subheader('Gain / Loss Curves')
        #fig, ax = plt.subplots(figsize =(10, 6))
        #CURVES.plot(ax=ax, x='threshold', y=['EARNED', 'LOST'])
        #plt.axvline(x=default_proba, color='red', linestyle='--', linewidth=2, alpha=0.5)
        #st.pyplot(fig)

        # CURVES NEW
        risk = st.selectbox(
            'Sélectionnez votre niveau de risque acceptable:',
            ('Bas', 'Moyen', 'Elevé')
        )

        risk_medium_lower = 0.210
        risk_medium_upper = 0.360
        risk_band = 0.1

        risk_graph_title = 'Placement du client dans les seuils de risque'

        if risk == 'Moyen':
            risk_lower = risk_medium_lower
            risk_upper = risk_medium_upper
        elif risk == 'Bas':
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
        ax.set_ylim(-55000,0)
        ax.legend(loc='lower left')

        plt.axvline(x=default_proba, ymin=0, ymax=1, color='black', linestyle='--', linewidth=1, alpha=0.4)
        plt.axvline(x=risk_lower, ymin=0, ymax=1, color='green', linestyle='--', linewidth=1, alpha=0.4)
        plt.axvline(x=risk_upper, ymin=0, ymax=1, color='darkred', linestyle='--', linewidth=1, alpha=0.4)

        ax.annotate('seuil de risque bas',
                    xy=(risk_lower, -40000), xycoords='data', color='green',
                    xytext=(-15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    horizontalalignment='right', verticalalignment='bottom')

        ax.annotate('seuil de risque elevé',
                    xy=(risk_upper, -40000), xycoords='data', color='darkred',
                    xytext=(15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='darkred', shrink=0.05),
                    horizontalalignment='left', verticalalignment='bottom')

        ax.annotate('probabilité de défaut du client',
                    xy=(default_proba, -30000), xycoords='data', color='black',
                    xytext=(-15, 25), textcoords='offset points',
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='right', verticalalignment='bottom')

        plt.suptitle(risk_graph_title)
        st.pyplot(fig, matplotlib=True)

        # in which category is the client regarding the currently selected risk?
        if default_proba <= risk_lower:
            risk_client = 'Low'
        elif default_proba >= risk_upper:
            risk_client = 'High'
        else:
            risk_client = 'Medium'
        base_text = 'La ligne verte correspond au seuil de risque faible. \
                    Au-dessus de la ligne rouge se trouve le seuil de risque élevé. \
                    Entre les lignes vertes et rouges se trouve une bande de risque moyen. \
                    \n\nLa ligne noire représente la probabilité de défaut calculée du client.\n\n'
        if risk_client == 'Medium':
            extra_text = 'Ce client tombe dans la tranche de risque **moyen** et vous devriez \
                étudier les valeurs du client pour les caractéristiques les plus importantes comme indiqué dans \
                les 2ème et 3ème onglets, pour tirer une conclusion manuelle sur l\'accord ou non du prêt.'
        elif risk_client == 'Low':
            extra_text = 'Ce client tombe dans la tranche de risque **faible** et pourrait être \
                **accepté** sans autre analyse manuelle, mais veuillez prendre un moment pour \
                étudier les valeurs du client pour les caractéristiques les plus importantes, comme indiqué dans \
                les 2e et 3e onglets, pour vous assurer que vous êtes d\'accord avec cette recommandation.'
        else: # risk == high:
            extra_text = 'Ce client tombe dans la tranche de risque **élevé** et pourrait être \
                rejeté sans autre analyse manuelle, mais veuillez prendre un moment pour \
                étudier les valeurs du client pour les caractéristiques les plus importantes comme indiqué dans \
                les 2ème et 3ème onglets, pour s\'assurer que vous êtes d\'accord avec cette recommandation.'
        risk_explanation_text = base_text + extra_text
        st.success(risk_explanation_text)



# TAB 2
with tab2:
    st.subheader('Analyse bivariée - Valeurs réelles')
    df_shap_values = pd.DataFrame(data=shap_values[1],columns=df.drop(columns=['TARGET']).columns)
    df_feature_importance = pd.DataFrame(columns=['feature','importance'])
    for col in df_shap_values.columns:
        importance = df_shap_values[col].abs().mean()
        df_feature_importance.loc[len(df_feature_importance)] = [col,importance]
    df_feature_importance = df_feature_importance.sort_values('importance',ascending=False)
    top_10_features = df_feature_importance.feature.head(20).to_list()
    default_ix = top_10_features.index('EXT_SOURCE_3')
    tab2_left_column, tab2_right_column = st.columns([1, 1], gap="small")
    with tab2_left_column:
        featureX = st.selectbox('Sélectionnez X:', top_10_features)
    with tab2_right_column:
        featureY = st.selectbox('Sélectionnez Y:', top_10_features, index=default_ix)
    if featureX == featureY:
        st.warning('**Attention -** Les X et Y sont les mêmes, le KDE ne sera pas affiché.')
    # scatter of the two features
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1,1,1)
    # add single point for the chosen client
    df2 = df.index.to_frame().reset_index(drop=True)
    pos_num = df2[df2['SK_ID_CURR'] == int(client_id)].index
    cm = df.TARGET.map({0: 'green', 1: 'red'})
    p = {0: "green", 1: "red"}
    ax.scatter(data=df, x=featureX, y=featureY, s=0.5, c=cm) # c='yellow'
    if featureX != featureY:
        sb.kdeplot(data=df, x=featureX, y=featureY, hue='TARGET', palette=p, alpha=0.3, fill=True)
    ax.scatter(data=df.iloc[pos_num], x=featureX, y=featureY, c='black', s=20, marker='o')
    ax.annotate('CLIENT(E)',
                xy=(df.iloc[pos_num][featureX], df.iloc[pos_num][featureY]), xycoords='data', color='black',
                xytext=(-25, 25), textcoords='offset points',
                arrowprops=dict(width=3, facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='bottom')
    ax.set_xlabel(featureX)
    ax.set_ylabel(featureY)
    custom = [
                Line2D([], [], marker='.', markersize=15, color='green', linestyle='None'),
                Line2D([], [], marker='.', markersize=15, color='red', linestyle='None'),
                Line2D([], [], marker='.', markersize=15, color='black', linestyle='None')
            ]
    fig.subplots_adjust(right=0.8)
    plt.title('Valeurs réelles, Client(e) ' + client_id)
    ax.legend(custom, ['Aucun défaut', 'Défaut','Client(e)'], loc='center left', fontsize=12, bbox_to_anchor=(0.8, 0.5), bbox_transform=fig.transFigure)
    st.pyplot(fig)

    # NEW LAYOUT
    st.subheader('Analyse bivariée - l\'importance')
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
    fig.suptitle('SHAP Feature Values, Client(e) ' + client_id)
    st.pyplot(fig)

    st.success('**Distributions des valeurs des caractéristiques et d\'importance** \
        \n\nDans le graphique du haut - Valeurs réelles, nous pouvons voir le placement du client  \
        dans la population en fonction des valeurs des deux caractéristiques sélectionnées. \
        \n\nDans le graphique du bas, nous voyons l\'importance de ces caractéristiques sur la prédiction \
        du modèle pour ce client. Une valeur d\'importance supérieure à zéro augmentera positivement  \
        la probabilité d\'un client défaillant. Une valeur inférieure à 0 diminuera cette probabilité. \
        \n\nPar conséquent, pour détecter un client potentiellement défaillant, il convient d\'accorder  \
        plus d\'importance à une paire de caractéristiques apparaissant en haut à droite de la  \
        distribution qu\'en bas à gauche. \
        \n\nLes distributions individuelles pour chaque caractéristique sélectionnée peuvent être  \
        vues à droite et en dessous de la distribution combinée.')

# TAB 3 - 4 HISTOGRAMS 4 FEATURES
with tab3:
    tab3_left_column, tab3_right_column = st.columns([1, 1], gap="small")
    with tab3_left_column:
        st.subheader('EXT_SOURCE_2')
        st.markdown('Valeur réelle: <span style="color:red">**' + str(round(EXT_SOURCE_2,4)) + '**</span>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="EXT_SOURCE_2", fill=True, color='blue').set(title='Dist EXT_SOURCE_2')
        plt.axvline(x=EXT_SOURCE_2, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)
        #plot_shap_summary_feature(shap_values[1],X,'EXT_SOURCE_2')
        #pos = get_feature_index('EXT_SOURCE_2',X)
        #fig = shap.summary_plot(shap_values[1][:,pos:pos+1], X.iloc[:, pos:pos+1])
        #st.pyplot(fig, matplotlib=True)

        st.subheader('EXT_SOURCE_3')
        st.markdown('Valeur réelle: <span style="color:red">**' + str(round(EXT_SOURCE_3,4)) + '**</span>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="EXT_SOURCE_3", fill=True, color='orange').set(title='Dist EXT_SOURCE_3')
        plt.axvline(x=EXT_SOURCE_3, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)
    # END TAB2 LEFT COLUMN

    # RIGHT COLUMN
    with tab3_right_column:
        st.subheader('PAYMENT_RATE')
        st.markdown('Valeur réelle: <span style="color:red">**' + str(round(PAYMENT_RATE,4)) + '**</span>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="PAYMENT_RATE", fill=True, color='green').set(title='Dist PAYMENT_RATE')
        plt.axvline(x=PAYMENT_RATE, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)

        st.subheader('AMT_CREDIT')
        st.markdown('Valeur réelle: <span style="color:red">**' + str(round(AMT_CREDIT,4)) + '**</span>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x="AMT_CREDIT", fill=True, color='yellow').set(title='Dist AMT_CREDIT')
        plt.axvline(x=AMT_CREDIT, color='red', linestyle='--', linewidth=2, alpha=0.5)
        st.pyplot(fig)
    # END TAB2 RIGHT COLUMN

        
