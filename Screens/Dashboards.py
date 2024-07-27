import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.plot_utils import read_df, build_dataframe

def build_page():
    build_header()
    build_body()

def build_header():
    text='<h1>Gráficos com base de diabetes</h1>'+\
    '<p>Esta página apresenta informações a partir da seguinte base de dados importada do '+\
    '<a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset">Kaggle</a>¹ .</p>'
    st.write(text, unsafe_allow_html=True)

def build_body():
    df = read_df()
    build_dataframe(df)
    st.markdown('<h2>Gráficos iniciais</h2>', unsafe_allow_html=True)
    build_diabetesplot_section(df)
    

def build_diabetesplot_section(df:pd.DataFrame):
    st.markdown('<h3>Histogramas diabéticos</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([.3,.7])
    cols = ['Fumante','Cobertura_de_saúde','Atividades_físicas']
    selec_col = c1.selectbox('Seleção', options=cols, key='selec_1')
    #stacked = c1.checkbox('Empilhado', value=True)
    #if stacked:
    fig = create_histograma_stacked(df, selec_col)
    #else:
    #    fig = create_histograma_stacked(df, selec_col)
    fig.update_layout(title=f'Histograma de {selec_col} por condição diabética.',
                      legend_title_text=selec_col, xaxis_title_text='Condição', yaxis_title_text='Quantidade')
    c2.plotly_chart(fig, use_container_width=True)
    
def create_histograma_stacked(df:pd.DataFrame, series_col:str) -> go.Figure:
    df = df.query(f'{series_col}.notna()').copy()
    return px.histogram(df, x='Diabetes', nbins=20, color=series_col, opacity=.75)