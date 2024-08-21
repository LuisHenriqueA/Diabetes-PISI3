import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.plot_utils import read_df  # Atualize se necessário
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

color_scale = ['#00ccff','#cc00ff','#ffcc00','#0066bb','#6600bb','#bb0066','#bb6600','#ff0066','#66ff66','#ee0503']
n_clusters = 3
clustering_cols = ['Smoker', 'HvyAlcoholConsump', 'Fruits', 'Veggies', 'Diabetes_012', 'BMI']

def build_page():
    build_header()
    build_body()

def build_header():
    st.write('<h1>Agrupamento (<i>Clustering</i>)</h1>', unsafe_allow_html=True)
    #st.write('''<i>Inicial.</i>''', unsafe_allow_html=True)

def build_body():
    global n_clusters
    c1, c2 = st.columns(2)
    # Remove o multiselect e define as colunas de clustering diretamente
    st.write("Colunas de clustering: " + ", ".join(clustering_cols))
    n_clusters = c2.slider('Quantidade de Clusters', min_value=2, max_value=10, value=3)
    df_raw = create_df_raw()
    df_clusters = create_df_clusters(df_raw)
    plot_cluster(df_clusters, 'cluster', 'Clusters')

def create_df_raw():
    cols = clustering_cols
    df_raw = pd.read_parquet('KDD/dfCleaned.parquet')  # Certifique-se de que esta função lê o seu dataframe corretamente
    df_raw = df_raw[cols].copy()  # Mantendo a leitura das colunas fixas
    return df_raw

def create_df_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df_clusters = df.copy()
    df_clusters['cluster'] = clusterize(df)
    return df_clusters

def clusterize(df: pd.DataFrame) -> pd.Series:
    X = df.values
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=4294967295)
    return kmeans.fit_predict(X)

def plot_cluster(df: pd.DataFrame, cluster_col: str, cluster_name: str):
    df[cluster_col] = df[cluster_col].apply(lambda x: f'Cluster {x}')
    st.write(f'<h3>{cluster_name}</h3>', unsafe_allow_html=True)
    st.dataframe(df)
    
    # Gráficos de barras
    st.write("### Distribuição das Colunas por Cluster")
    cluster_summary = df.groupby(cluster_col)[clustering_cols].mean()
    
    for c in clustering_cols:
        fig = px.bar(cluster_summary, x=cluster_summary.index, y=c,
                     color=cluster_summary.index, color_discrete_sequence=color_scale,
                     labels={c: f'Média de {c}'})
        st.plotly_chart(fig, use_container_width=True)