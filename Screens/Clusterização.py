import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import numpy as np

# Configurações
color_scale = ['#00ccff', '#cc00ff', '#ffcc00', '#0066bb', '#6600bb', '#bb0066', '#bb6600', '#ff0066', '#66ff66', '#ee0503']
n_clusters = 3
clustering_cols = ['Smoker', 'HvyAlcoholConsump', 'Fruits', 'Veggies', 'Diabetes_012', 'BMI']

def build_page():
    build_header()
    build_body()

def build_header():
    st.write('<h1>Agrupamento com K-means (<i>Clustering</i>)</h1>', unsafe_allow_html=True)

def build_body():
    global n_clusters
    c1, c2 = st.columns(2)
    st.write("Colunas de clustering: " + ", ".join(clustering_cols))
    n_clusters = c2.slider('Quantidade de Clusters', min_value=2, max_value=10, value=3)
    df_raw = create_df_raw()
    
    if df_raw.empty:
        st.write("O DataFrame de dados brutos está vazio. Verifique a origem dos dados.")
        return

    min_bmi, max_bmi = st.slider('Selecione o intervalo de BMI', 
                                 min_value=float(df_raw['BMI'].min()), 
                                 max_value=float(df_raw['BMI'].max()), 
                                 value=(float(df_raw['BMI'].min()), float(df_raw['BMI'].max())),
                                 step=0.1)
    
    df_filtered = df_raw[(df_raw['BMI'] >= min_bmi) & (df_raw['BMI'] <= max_bmi)]
    
    if df_filtered.empty:
        st.write("Nenhuma amostra no intervalo de BMI selecionado.")
        return
    
    df_clusters = create_df_clusters(df_filtered)
    plot_cluster(df_clusters, 'cluster', 'Clusters')
    plot_lifestyle_disparities(df_clusters)
    
    feature = st.selectbox("Selecione a feature para o gráfico de lollipop", options=[col for col in df_raw.columns if col != 'cluster'])
    plot_lollipop_cluster_vs_feature(df_clusters, feature)
    plot_bubble_clusters_vs_diabetes(df_clusters)
    plot_boxplot_bmi_clusters(df_clusters)

def create_df_raw():
    cols = clustering_cols
    try:
        df_raw = pd.read_parquet('KDD/dfCleaned.parquet')  # Certifique-se de que esta função lê o seu dataframe corretamente
        df_raw = df_raw[cols].copy()
        return df_raw
    except FileNotFoundError:
        st.write("Arquivo 'KDD/dfCleaned.parquet' não encontrado.")
        return pd.DataFrame()  # Retorna um DataFrame vazio em caso de erro

def create_df_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df_clusters = df.copy()
    df_clusters['cluster'] = clusterize(df)
    return df_clusters

def clusterize(df: pd.DataFrame) -> pd.Series:
    X = df.values
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=4294967295)
    return kmeans.fit_predict(X)

def plot_cluster(df: pd.DataFrame, cluster_col: str, cluster_name: str):
    if df.empty:
        st.write("O DataFrame está vazio. Não é possível gerar gráficos.")
        return

    df[cluster_col] = df[cluster_col].apply(lambda x: f'Cluster {x}')
    st.write(f'<h3>{cluster_name}</h3>', unsafe_allow_html=True)
    st.dataframe(df)

    st.write("### Distribuição das Colunas por Cluster")
    cluster_summary = df.groupby(cluster_col)[clustering_cols].mean()

    for c in clustering_cols:
        fig = px.bar(cluster_summary, x=cluster_summary.index, y=c,
                     color=cluster_summary.index, color_discrete_sequence=color_scale,
                     labels={c: f'Média de {c}'})
        st.plotly_chart(fig, use_container_width=True)

def plot_lifestyle_disparities(df: pd.DataFrame):
    st.write("### Disparidades de Hábitos de Vida entre Pacientes Não Diabéticos e Diabéticos")

    if df.empty:
        st.write("O DataFrame está vazio. Verifique os dados carregados.")
        return

    diabetes_options = [0, 1]
    df_filtered = df[df['Diabetes_012'].astype(int).isin(diabetes_options)]

    if df_filtered.empty:
        st.write("O DataFrame filtrado está vazio. Verifique a filtragem dos dados.")
        return

    lifestyle_cols = ['Smoker', 'HvyAlcoholConsump', 'Fruits', 'Veggies']

    for col in lifestyle_cols:
        if col in df_filtered.columns:
            distribution = df_filtered.groupby('Diabetes_012')[col].value_counts(normalize=True).unstack().fillna(0)
            
            fig = go.Figure()
            for value in distribution.columns:
                fig.add_trace(go.Bar(
                    x=distribution.index,
                    y=distribution[value],
                    name=f'{col} - {value}',
                    marker_color=color_scale[0] if value == 1 else color_scale[1]
                ))
            fig.update_layout(
                barmode='stack',
                title=f'Disparidades em {col} por Status de Diabetes',
                xaxis_title='Status de Diabetes',
                yaxis_title=f'Distribuição de {col}',
                xaxis=dict(tickvals=[0, 1], ticktext=['Não Diabético', 'Diabético'])
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"A coluna {col} não está presente no DataFrame.")

def plot_lollipop_cluster_vs_feature(df: pd.DataFrame, feature: str):
    st.write(f"### Lollipop Chart: Cluster vs {feature}")

    if df.empty:
        st.write("O DataFrame está vazio. Não é possível gerar gráficos.")
        return

    if 'cluster' not in df.columns or feature not in df.columns:
        st.write(f"As colunas 'cluster' e/ou '{feature}' não estão presentes no DataFrame.")
        return

    if df[feature].dtype == 'object':
        grouped_counts = df.groupby(['cluster', feature]).size().reset_index(name='counts')
        feature_values = df[feature].unique()
        traces = []
        for value in feature_values:
            value_data = grouped_counts[grouped_counts[feature] == value]
            value_data = value_data.sort_values(by='counts', ascending=True)
            value_data['position'] = np.arange(len(value_data))
            traces.append(go.Scatter(
                x=value_data['counts'], 
                y=value_data['position'],
                mode='markers',
                marker=dict(size=10),
                name=f'{value}',
                text=value_data.apply(lambda row: f"Cluster {row['cluster']} - {value}: {row['counts']}", axis=1),
                hoverinfo='text'
            ))
            for i in range(len(value_data)):
                traces.append(go.Scatter(
                    x=[0, value_data['counts'].iloc[i]], 
                    y=[value_data['position'].iloc[i], value_data['position'].iloc[i]],
                    mode='lines',
                    line=dict(width=2),
                    showlegend=False
                ))
    else:
        grouped_means = df.groupby('cluster')[feature].mean().reset_index(name='mean_value')
        grouped_means = grouped_means.sort_values(by='mean_value', ascending=True)
        grouped_means['position'] = np.arange(len(grouped_means))
        traces = [
            go.Scatter(
                x=grouped_means['mean_value'], 
                y=grouped_means['position'],
                mode='markers',
                marker=dict(size=10),
                name=feature,
                text=grouped_means.apply(lambda row: f"Cluster {row['cluster']} - Média: {row['mean_value']:.2f}", axis=1),
                hoverinfo='text'
            )
        ]
        for i in range(len(grouped_means)):
            traces.append(go.Scatter(
                x=[0, grouped_means['mean_value'].iloc[i]], 
                y=[grouped_means['position'].iloc[i], grouped_means['position'].iloc[i]],
                mode='lines',
                line=dict(width=2),
                showlegend=False
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Lollipop Chart: Cluster vs {feature}',
        xaxis_title='Count' if df[feature].dtype == 'object' else 'Mean Value',
        yaxis_title='Cluster',
        yaxis=dict(
            tickmode='array',
            tickvals=grouped_means['position'] if df[feature].dtype != 'object' else np.concatenate([value_data['position'] for value in feature_values]),
            ticktext=grouped_means['cluster'].astype(str) if df[feature].dtype != 'object' else np.concatenate([value_data['cluster'].astype(str) for value in feature_values])
        ),
        showlegend=True,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_bubble_clusters_vs_diabetes(df: pd.DataFrame):
    st.write("### Gráfico de Bolhas: Clusters vs Status de Diabetes")

    if df.empty:
        st.write("O DataFrame está vazio. Não é possível gerar gráficos.")
        return

    if 'cluster' not in df.columns or 'Diabetes_012' not in df.columns:
        st.write("As colunas 'cluster' e/ou 'Diabetes_012' não estão presentes no DataFrame.")
        return

    df_counts = df.groupby(['cluster', 'Diabetes_012']).size().reset_index(name='counts')

    fig = px.scatter(df_counts, x='cluster', y='Diabetes_012', size='counts', color='cluster',
                     color_discrete_sequence=color_scale[:n_clusters],
                     labels={'cluster': 'Cluster', 'Diabetes_012': 'Status de Diabetes', 'counts': 'Contagem'},
                     title='Gráfico de Bolhas: Clusters vs Status de Diabetes')

    fig.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Status de Diabetes',
        legend_title='Cluster'
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_boxplot_bmi_clusters(df: pd.DataFrame):
    st.write("### Distribuição de BMI por Clusters")

    # Verifica se as colunas 'BMI' e 'cluster' estão presentes no DataFrame
    if 'BMI' not in df.columns or 'cluster' not in df.columns:
        st.write("As colunas 'BMI' e/ou 'cluster' não estão presentes no DataFrame.")
        return

    u = df[['BMI', 'cluster']]

    # Cria o gráfico boxplot
    fig = go.Figure()

    # Itera sobre os clusters para adicionar uma trace para cada cluster
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = u.loc[u['cluster'] == cluster, 'BMI']
        
        fig.add_trace(go.Box(
            y=cluster_data,
            name=f'Cluster {cluster}',
            boxmean='sd',  # Adiciona a média e o desvio padrão
        ))

    fig.update_layout(
        title='Distribuição de BMI por Clusters',
        xaxis_title='Cluster',
        yaxis_title='Valores de BMI',
        boxmode='group',  # Agrupa as caixas
    )

    st.plotly_chart(fig, use_container_width=True)


