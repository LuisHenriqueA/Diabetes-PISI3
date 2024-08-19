import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.plot_utils import read_df  # Atualize se necessário
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

color_scale = ['#00ccff','#cc00ff','#ffcc00','#0066bb','#6600bb','#bb0066','#bb6600','#ff0066','#66ff66','#ee0503']
n_clusters = 3
clustering_cols_opts = ['Fumante_val', 'HvyAlcoholConsump', 'Fruits', 'Veggies', 'Saúde_geral_val', 'MentHlth', 'Diabetes_val', 'IMC']
clustering_cols = clustering_cols_opts.copy()

def build_page():
    build_header()
    build_body()

def build_header():
    st.write('<h1>Agrupamento (<i>Clustering</i>)</h1>', unsafe_allow_html=True)
    st.write('''<i>Inicial.</i>''', unsafe_allow_html=True)

def build_body():
    global n_clusters, clustering_cols
    c1, c2 = st.columns(2)
    clustering_cols = c1.multiselect('Colunas', options=clustering_cols_opts, default=clustering_cols_opts[0:2])
    if len(clustering_cols) < 2:
        st.error('É preciso selecionar pelo menos 2 colunas.')
        return
    n_clusters = c2.slider('Quantidade de Clusters', min_value=2, max_value=10, value=3)
    df_raw = create_df_raw()
    df_clusters = create_df_clusters(df_raw)
    plot_cluster(df_clusters, 'cluster', 'Cluster Sem Normalização')

def create_df_raw():
    cols = ['Fumante_val', 'HvyAlcoholConsump', 'Fruits', 'Veggies', 'Saúde_geral_val', 'MentHlth', 'Diabetes_val', 'IMC']
    df_raw = read_df()  # Certifique-se de que esta função lê o seu dataframe corretamente
    df_raw = df_raw[cols].copy()  # Mantendo a leitura das colunas selecionadas pelo usuário
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
    df.sort_values(by=[cluster_col, 'Diabetes_val'], inplace=True)
    df[cluster_col] = df[cluster_col].apply(lambda x: f'Cluster {x}')
    st.write(f'<h3>{cluster_name}</h3>', unsafe_allow_html=True)
    st.dataframe(df)
    
    # Gráficos de dispersão
    cols = st.columns(len(clustering_cols))
    for c1 in clustering_cols:
        for cidx, c2 in enumerate(clustering_cols):
            fig = px.scatter(df, x=c1, y=c2, color=cluster_col, color_discrete_sequence=color_scale)
            cols[cidx].plotly_chart(fig, use_container_width=True)
    
    # Gráficos de barras
    st.write('<h3>Distribuição dos Dados Binários por Cluster</h3>', unsafe_allow_html=True)
    for col in clustering_cols:
        if df[col].dtype in [int, float]:  # Verifica se a coluna é numérica
            cluster_counts = df.groupby([cluster_col])[col].sum().reset_index()
            cluster_counts[col] = cluster_counts[col]  # Contagem de 1s
            cluster_counts = cluster_counts.rename(columns={col: 'Count'})
            fig = px.bar(cluster_counts, x=cluster_col, y='Count', color=cluster_col, color_discrete_sequence=color_scale,
                        labels={cluster_col: 'Cluster', 'Count': f'Count of {col}'})
            st.plotly_chart(fig, use_container_width=True)
build_page()