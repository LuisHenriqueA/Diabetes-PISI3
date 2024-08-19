import pandas as pd
import streamlit as st
from enum import Enum
from KDD.classifier import ClassifierType

def initialize_session_state():
    if 'DiffWalk' not in st.session_state:
        st.session_state['DiffWalk'] = 'DiffWalk'

def build_page():
    initialize_session_state()
    build_header()
    build_body()

def build_header():
    st.write('<h1>Classificação com Machine Learning</h1>', unsafe_allow_html=True)

def build_body():
    df = load_df()
    build_controls(df)
    balance = st.checkbox('Balancear o conjunto de dados', value=False)
    Classificador_sel = st.session_state['classificador']
    classe_sel = st.session_state['DiffWalk']
    caracteristicas_sel = st.session_state['caracteristicas']
    df = df[caracteristicas_sel + [classe_sel]]
    Classifier_Type = ClassifierType.get(Classificador_sel)
    Classificador = Classifier_Type.\
        build(df, classe_sel, balance=balance)
    Classificador.classify()

def build_controls(df):
    all_columns = df.columns.tolist()
    c1, c2 = st.columns([.3, .7])
    class_cols = ['DiffWalk']
    class_col = c1.selectbox('Target', options=class_cols,  index=len(class_cols)-1, key='classe')
    features_opts = [col for col in all_columns]
    features = features_opts.copy()
    features.remove(class_col)
    features = c2.multiselect('Características *(Features)*', options=features,  default=features, key='caracteristicas')
    if len(features) < 2:
        st.error('É preciso selecionar pelo menos 2 características.')
        return
    c1, c2, c3 = st.columns(3)
    c1.selectbox('Classificador', options=ClassifierType.values(), index=0, key='classificador')

def load_df()->pd.DataFrame:
    df_raw = _ingest_df()
    return df_raw

def _ingest_df()->pd.DataFrame:
    return pd.read_csv('KDD/dfCleaned.csv')