import os
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
from matplotlib import colors


class ClusterAnalysisApp:
    def __init__(self, data_path):
        """
        Inicializa a aplicação com o caminho para o arquivo de dados.
        """
        self.data_path = data_path
        self.df = None
        self.binary_columns = [
            'Sex', 'Smoker', 'HvyAlcoholConsump', 'PhysActivity', 'Fruits', 'Veggies', 
            'HighBP', 'HighChol', 'Stroke', 'HeartDiseaseorAttack', 'DiffWalk'
        ]
        self.numeric_columns = ['BMI', 'Diabetes_012']  # Incluindo variáveis numéricas adicionais
        self.color_map = {0: '#0BAB7C', 1: '#C7F4C2'}  # Novas cores para os clusters

    def load_data(self):
        """
        Carrega o dataset do arquivo Parquet.
        """
        if os.path.exists(self.data_path):
            self.df = pd.read_parquet(self.data_path)
        else:
            st.error(f"Arquivo não encontrado: {self.data_path}")
            st.stop()

    def plot_bar_charts(self, df_filtered, selected_vars):
        """
        Cria gráficos de barras para variáveis binárias por cluster.
        """
        for column in selected_vars:
            st.write(f'Distribuição para {column}')
            st.markdown("Gráfico de barras mostrando a distribuição de valores binários por cluster.")
            fig = px.histogram(
                df_filtered, 
                x=column, 
                color='cluster', 
                barmode='group', 
                title=f'Distribuição de {column} por Cluster',
                color_discrete_map=self.color_map
            )
            st.plotly_chart(fig)

    def plot_heatmap(self, df_filtered):
        """
        Cria um heatmap de frequência das variáveis binárias por cluster.
        """
        st.subheader('Heatmap de Frequência das Variáveis Binárias')
        st.markdown("Este heatmap mostra a frequência média de cada variável binária dentro de cada cluster.")

        heatmap_data = df_filtered[self.binary_columns + ['cluster']]
        heatmap_data = heatmap_data.groupby('cluster').mean().T

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis')
        st.pyplot(plt.gcf())

    def plot_radar_chart(self, df_filtered, cluster):
        """
        Cria um gráfico de radar para múltiplas variáveis binárias de um cluster.
        """
        data = df_filtered[df_filtered['cluster'] == cluster]
        mean_values = data[self.binary_columns].mean().values
        categories = list(self.binary_columns)

        mean_values = np.concatenate((mean_values, [mean_values[0]]))
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        if cluster in self.color_map:
            color = self.color_map[cluster]
        else:
            color = '#333333'  # Cor padrão se o cluster não estiver no dicionário de cores

        ax.plot(angles, mean_values, linewidth=1, linestyle='solid', label=f'Cluster {cluster}', color=color)
        ax.fill(angles, mean_values, alpha=0.3, color=color)
        ax.set_ylim(0, 1)

        plt.xticks(angles[:-1], categories, color='grey', size=8)
        plt.title(f'Perfil do Cluster {cluster}')
        ax.grid(True)

        st.pyplot(plt.gcf())

    def plot_boxplot(self, df_filtered):
        """
        Cria um boxplot para visualizar a distribuição de variáveis numéricas por cluster.
        """
        st.subheader('Boxplot de Variáveis Numéricas por Cluster')
        st.markdown("O boxplot visualiza a distribuição e outliers das variáveis numéricas dentro de cada cluster.")
        for var in self.numeric_columns:
            fig = px.box(df_filtered, x='cluster', y=var, color='cluster', 
                         title=f'Boxplot de {var} por Cluster', color_discrete_map=self.color_map)
            st.plotly_chart(fig)

    def plot_scatter(self, df_filtered):
        """
        Cria um scatter plot para explorar a relação entre duas variáveis selecionadas.
        """
        st.subheader('Scatter Plot de Variáveis Selecionadas')
        st.markdown("Scatter plot para explorar a correlação entre duas variáveis numéricas selecionadas.")
        selected_x = st.selectbox('Selecione a variável X', self.numeric_columns)
        selected_y = st.selectbox('Selecione a variável Y', self.numeric_columns)

        if selected_x and selected_y:
            fig = px.scatter(df_filtered, x=selected_x, y=selected_y, color='cluster',
                             title=f'Scatter Plot de {selected_x} vs {selected_y} por Cluster', 
                             color_discrete_map=self.color_map)
            st.plotly_chart(fig)

    def plot_histogram(self, df_filtered):
        """
        Cria um histograma interativo para explorar a distribuição de uma variável numérica selecionada.
        """
        st.subheader('Histograma de Variável Numérica')
        st.markdown("Histograma mostrando a distribuição de uma variável numérica por cluster.")
        selected_var = st.selectbox('Selecione uma Variável para o Histograma', self.numeric_columns)

        if selected_var:
            fig = px.histogram(df_filtered, x=selected_var, color='cluster', barmode='overlay', 
                               title=f'Histograma de {selected_var} por Cluster', 
                               color_discrete_map=self.color_map)
            st.plotly_chart(fig)

    def build_page(self):
        """
        Método principal para construir a interface do Streamlit.
        """
        st.title('Análise Interativa de Clusters')

        # Carregar dados
        self.load_data()

        # Seleção de Cluster
        clusters = self.df['cluster'].unique()
        selected_cluster = st.multiselect('Selecione o Cluster para Visualização', clusters, default=clusters)

        # Filtrando o DataFrame com base na seleção do usuário
        df_filtered = self.df[self.df['cluster'].isin(selected_cluster)]

        # Filtragem de variáveis para gráfico de barras
        st.subheader('Seleção de Variáveis Binárias')
        selected_vars = st.multiselect('Selecione as Variáveis Binárias para Visualizar', self.binary_columns, default=self.binary_columns)

        # Filtro para seleção de gráficos
        st.subheader('Seleção de Gráficos')
        available_plots = {
            'Distribuição das Variáveis Binárias por Cluster': 'plot_bar_charts',
            'Heatmap de Frequência': 'plot_heatmap',
            'Gráfico Radar para Análise de Múltiplas Variáveis Binárias': 'plot_radar_chart',
            'Boxplot de Variáveis Numéricas por Cluster': 'plot_boxplot',
            'Scatter Plot de Variáveis Selecionadas': 'plot_scatter',
            'Histograma de Variável Numérica': 'plot_histogram'
        }
        selected_plots = st.multiselect('Selecione os Gráficos que Deseja Visualizar', list(available_plots.keys()), default=list(available_plots.keys()))

        # Gráficos Interativos para Colunas Binárias
        if 'Distribuição das Variáveis Binárias por Cluster' in selected_plots and selected_vars:
            self.plot_bar_charts(df_filtered, selected_vars)

        # Heatmap de Frequência
        if 'Heatmap de Frequência' in selected_plots:
            self.plot_heatmap(df_filtered)

        # Radar Chart para Visualização de Múltiplas Variáveis Binárias
        if 'Gráfico Radar para Análise de Múltiplas Variáveis Binárias' in selected_plots:
            for cluster in selected_cluster:
                self.plot_radar_chart(df_filtered, cluster)

        # Boxplot de Variáveis Numéricas por Cluster
        if 'Boxplot de Variáveis Numéricas por Cluster' in selected_plots:
            self.plot_boxplot(df_filtered)

        # Scatter Plot para Variáveis Selecionadas
        if 'Scatter Plot de Variáveis Selecionadas' in selected_plots:
            self.plot_scatter(df_filtered)

        # Histograma de Variável Numérica
        if 'Histograma de Variável Numérica' in selected_plots:
            self.plot_histogram(df_filtered)


# Função para construir a página
def build_page():
    data_path = 'KDD/ClustResult/dfKmeans.parquet'
    app = ClusterAnalysisApp(data_path)
    app.build_page()
