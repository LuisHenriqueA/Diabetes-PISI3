import pandas as pd
import streamlit as st
#import plotly.express as px
from utils.df_functions import read_parquet
#from df_functions import read_csv

def __rename_data() -> pd.DataFrame:
    df=read_parquet('diabetes_012_health_indicators_BRFSS2015')
    df.rename(columns={
        'Diabetes_012':'Diabetes_val', 'HighBP':'Hipertensão_val', 'HighChol':'Colesterol_alto_val',
        'CholCheck':'Colesterol_checado', 'BMI':'IMC', 'Smoker':'Fumante_val',
        'PhysActivity':'Atividades_físicas_val', 'GenHlth':'Saúde_geral_val', 'Age':'Idade',
        'AnyHealthcare':'Cobertura_saúde_val', 'Sex':'Sexo_val'
    }, inplace=True)
    return df

def __transform_data(df:pd.DataFrame) -> pd.DataFrame:
    df['Diabetes'] = df['Diabetes_val'].map({
        0:'Não diabético', 1:'Pré-diabético', 2:'Diabético'
    })
    df['Fumante'] = df['Fumante_val'].map({
        0:'não', 1:'sim'
    })
    df['Sexo'] = df['Sexo_val'].map({
        0:'Feminino', 1:'Masculino'
    })
    df['Atividades_físicas'] = df['Atividades_físicas_val'].map({
        0:'não', 1:'sim'
    })
    df['Cobertura_de_saúde'] = df['Cobertura_saúde_val'].map({
        0:'não', 1:'sim'
    })
    df['Saúde_geral'] = df['Saúde_geral_val'].map({
        1:'Excelente', 2:'Muito boa', 3:'Boa', 4:'Razoável', 5:'Ruim'
    })
    df.sort_values(by=['Fumante','Atividades_físicas','Cobertura_de_saúde', 'Saúde_geral_val'], inplace=True)
    return df

def read_df() -> pd.DataFrame:
    return __transform_data(__rename_data())

def build_dataframe(df:pd.DataFrame):
    with st.expander('Dados do dataset'):
        st.dataframe(df)