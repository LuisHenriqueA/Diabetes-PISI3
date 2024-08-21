import pandas as pd
#import numpy as np
import streamlit as st
#import seaborn as sns
#import matplotlib.pyplot as plt
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
    df = read_df('KDD/dfCleaned')
    df_bruto = read_df('DiabetesDataSet/diabetes_012_health_indicators_BRFSS2015')
    build_dataframe(df)
    st.markdown('<h2>Gráficos iniciais</h2>', unsafe_allow_html=True)
    build_diabetesplot_section(df)
    build_boxpolot_expander(df, df_bruto)

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
    color_map = {'sim': '#0BAB7C', 'não': '#C7F4C2'}
    return px.histogram(df, x='Diabetes', nbins=20, color=series_col, opacity=.75, color_discrete_map=color_map)

def build_boxplot_section(df:pd.DataFrame):
    st.markdown('<h3>Diagramas de caixa</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([.3,.7])
    cols = ['IMC', 'Saúde_geral']
    selec_col = c1.selectbox('Seleção', options=cols, key='selec_2')
    reverse = c1.checkbox('Invertido', value=False)
    col_selected = [selec_col, 'Diabetes']
    if reverse:
        col_selected.reverse()
    df_plot = df[col_selected]
    fig = px.box(df_plot,x=col_selected[0], y=col_selected[1])
    fig.update_traces(marker_color='#0BAB7C')
    c2.plotly_chart(fig, use_container_width=True)

def build_boxpolot_expander(df: pd.DataFrame, df_bruto: pd.DataFrame):
    st.markdown('<h3>Gráficos Exploratórios</h3>', unsafe_allow_html=True)
    
    # Checkbox para selecionar o DataFrame
    use_bruto = st.checkbox('Adicionar outliers', value=False)
    selected_df = df_bruto if use_bruto else df
    
    cols = ['Diabetes', 'IMC', 'Saúde_geral']
    df_plot = selected_df[cols]
    
    # Filtrando para ignorar pré-diabéticos
    df_filtered = df_plot[df_plot['Diabetes'] != 'Pré-diabético']
    
    # Heatmap para condição diabética por saúde geral com cores ajustadas
    with st.expander('Heatmap de condição diabética por Saúde geral'):
        _, c2, _ = st.columns([1, 2, 1])
        # Criando uma tabela de frequência para o heatmap com os dados filtrados
        heatmap_data = df_filtered.pivot_table(index='Saúde_geral', columns='Diabetes', aggfunc='size', fill_value=0)
        fig1 = px.imshow(heatmap_data, 
                         labels=dict(x="Condição diabética", y="Saúde geral", color="Frequência"),
                         color_continuous_scale='YlOrRd')
        fig1.update_layout(xaxis_title_text='Condição diabética', yaxis_title_text='Saúde geral')
        c2.plotly_chart(fig1)
    
    # Boxplot de IMC por condição diabética
    with st.expander('Boxplot de IMC por condição diabética'):
        _, c2, _ = st.columns([1, 2, 1])
        fig2 = px.box(df_plot, x=cols[0], y=cols[1])
        fig2.update_traces(marker_color='#0BAB7C')
        fig2.update_layout(yaxis_title_text='IMC', xaxis_title_text='Condição diabética')
        c2.plotly_chart(fig2)