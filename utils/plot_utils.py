import pandas as pd
import streamlit as st
from utils.df_functions import read_parquet
from df_functions import read_csv

def __rename_data(df) -> pd.DataFrame:
    df=read_parquet(df)
    return df

def __transform_data(df:pd.DataFrame) -> pd.DataFrame:
    colunas_renomeadas = {
        'Sex': 'Sexo',
        'Smoker': 'Fumante',
        'HvyAlcoholConsump': 'Consumo excessivo de álcool',  # Você pode definir o nome que quiser aqui
        'PhysActivity': 'Pratica atividade física',
        'Fruits': 'Consome frutas diariamente',
        'Veggies': 'Consome vegetais diariamente',
        'HighBP': 'Pressão alta',
        'HighChol': 'Colesterol alto',
        'Stroke': 'AVC',
        'HeartDiseaseorAttack': 'Doença Coronariana ou Infarto do Miocárdio',
        'DiffWalk': 'Dificuldade em se locomover',
        'BMI': 'IMC',
        'Diabetes_012': 'Não, pré ou Diabético',
        'MentHlth': 'Saúde Mental',
        'GenHlth': 'Saúde geral',
        'AnyHealthcare': 'Cobertura de saúde',
        'PhysHlth': 'Saúde física',
    }

    # Renomeia as colunas do DataFrame
    df.rename(columns=colunas_renomeadas, inplace=True)
    return df

def read_df(df) -> pd.DataFrame:
    return __transform_data(__rename_data(df))

def build_dataframe(df:pd.DataFrame):
    with st.expander('Dados do dataset'):
        st.dataframe(df)