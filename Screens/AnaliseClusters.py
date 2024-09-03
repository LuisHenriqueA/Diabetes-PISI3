import os
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
from matplotlib import colors
import plotly.graph_objects as go


class ClusterAnalysisApp:
    def __init__(self, data_path):
        """
        Inicializa a aplicação com o caminho para o arquivo de dados.
        """
        self.data_path = data_path
        self.df = None
        self.binary_columns = [
            'Sexo', 'Fumante', 'Consumo excessivo de álcool', 'Pratica atividade física', 'Consome frutas diariamente', 
            'Consome vegetais diariamente', 'Pressão alta', 'Colesterol alto', 'AVC', 
            'Doença Coronariana ou Infarto do Miocárdio', 'Dificuldade em Subir Escadas'
        ]
        self.numeric_columns = ['IMC', 'Não, pré ou Diabético']  # Incluindo variáveis numéricas adicionais
        self.color_map = {0: '#0BAB7C', 1: '#C7F4C2'}  # Novas cores para os clusters

    def renomear_colunas(self, df):
        """
        Renomeia as colunas do DataFrame com base em um dicionário de mapeamento.
        """
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
            'DiffWalk': 'Dificuldade em Subir Escadas',
            'BMI': 'IMC',
            'Diabetes_012': 'Não, pré ou Diabético'
        }

        # Renomeia as colunas do DataFrame
        df.rename(columns=colunas_renomeadas, inplace=True)
        return df

    def load_data(self):
        """
        Carrega o dataset do arquivo Parquet e renomeia as colunas.
        """
        if os.path.exists(self.data_path):
            self.df = pd.read_parquet(self.data_path)
            # Renomear as colunas após carregar o DataFrame
            self.df = self.renomear_colunas(self.df)
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
                color_discrete_map={0: '#0BAB7C', 1: '#C7F4C2'},
                color_discrete_sequence=['#0BAB7C', '#C7F4C2']
            )
            fig.update_traces(marker=dict(line=dict(width=0)))  # Remove linhas em volta das barras para destacar as cores
            st.plotly_chart(fig)

    def plot_heatmap(self, df_filtered):
        """
        Cria um heatmap interativo de frequência das variáveis binárias por cluster,
        mostrando os valores diretamente nas células.
        """
        st.subheader('Heatmap de Frequência das Variáveis Binárias')
        st.markdown("Este heatmap mostra a frequência média de cada variável binária dentro de cada cluster.")

        # Filtrando as colunas binárias e a coluna de cluster
        heatmap_data = df_filtered[self.binary_columns + ['cluster']]
    
        # Agrupando por cluster e calculando a média
        heatmap_data = heatmap_data.groupby('cluster').mean().T

        # Criando o heatmap interativo usando Plotly
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Cluster", y="Variável", color="Frequência Média"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='greens',
            text_auto=True  # Adiciona os valores nas células
        )
    
        # Atualizando a formatação dos valores nas células
        fig.update_traces(texttemplate="%{z:.2f}", textfont_size=12)  # Usando o valor correto para referenciar os dados

        # Exibindo o gráfico no Streamlit
        st.plotly_chart(fig)

    def plot_radar_chart(self, df_filtered, cluster):
        """
        Cria um gráfico de radar interativo otimizado para múltiplas variáveis binárias de um cluster.
        """
        data = df_filtered[df_filtered['cluster'] == cluster]
        mean_values = data[self.binary_columns].mean().values
        categories = list(self.binary_columns)

        mean_values = np.concatenate((mean_values, [mean_values[0]]))  # Fechando o loop do radar

        fig = go.Figure(data=go.Scatterpolar(
            r=mean_values,
            theta=categories + [categories[0]],  # Fechando o loop do radar
            fill='toself',
            line_color='#0BAB7C',  # Cor da linha
            fillcolor='rgba(11, 171, 124, 0.4)',  # Cor de preenchimento com transparência
            opacity=0.6
        ))

        # Configurações simplificadas para otimizar a performance
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,  # Removendo a legenda para reduzir o tempo de renderização
            title=f'Perfil do Cluster {cluster}',
            margin=dict(l=30, r=30, t=30, b=30)  # Margens menores para otimizar o espaço
        )

        # Exibindo o gráfico interativo no Streamlit
        st.plotly_chart(fig, use_container_width=True)  # Tenta otimizar o uso de largura

    def plot_boxplot(self, df_filtered):
        """
        Cria um boxplot para visualizar a distribuição de variáveis numéricas por cluster.
        """
        st.subheader('Boxplot de Variáveis Numéricas por Cluster')
        st.markdown("O boxplot visualiza a distribuição e outliers das variáveis numéricas dentro de cada cluster.")
        
        fig = px.box(df_filtered, x='cluster', y=self.numeric_columns[0], color='cluster', 
                         title=f'Boxplot de IMC por Cluster', color_discrete_map=self.color_map, color_discrete_sequence=['#0BAB7C', '#C7F4C2'])
        st.plotly_chart(fig)

    def plot_scatter(self, df_filtered):
        """
        Cria um scatter plot para explorar a relação entre duas variáveis selecionadas.
        """
        st.subheader('Scatter Plot de Variáveis Selecionadas')
        st.markdown("Scatter plot para explorar a correlação entre duas variáveis numéricas selecionadas.")
        col1, col2 = st.columns(2)
        with col1:
            selected_x = st.selectbox('Selecione a variável X', self.numeric_columns, key='scatter_x')
        with col2:
            selected_y = st.selectbox('Selecione a variável Y', self.numeric_columns, key='scatter_y')

        if selected_x and selected_y:
            fig = px.scatter(df_filtered, x=selected_x, y=selected_y, color='cluster',
                             title=f'Scatter Plot de {selected_x} vs {selected_y} por Cluster', 
                             color_discrete_map=self.color_map, color_discrete_sequence=['#0BAB7C', '#C7F4C2'])
            st.plotly_chart(fig)

    def plot_histogram(self, df_filtered):
        """
        Cria um histograma interativo para explorar a distribuição de uma variável numérica selecionada.
        """
        st.subheader('Histograma de Variável Numérica')
        st.markdown("Histograma mostrando a distribuição de uma variável numérica por cluster.")
        selected_var = st.selectbox('Selecione uma Variável para o Histograma', self.numeric_columns, key='histogram_var')

        if selected_var:
            fig = px.histogram(df_filtered, x=selected_var, color='cluster', barmode='overlay', 
                               title=f'Histograma de {selected_var} por Cluster', 
                               color_discrete_map=self.color_map,
                               opacity=0.7, color_discrete_sequence=['#0BAB7C', '#C7F4C2'])
            st.plotly_chart(fig)


    def plot_pareto(self, df_filtered):
        """
        Cria um gráfico de Pareto para mostrar a importância das variáveis binárias na formação dos clusters.
        """
        st.subheader('Gráfico de Pareto para Análise de Importância de Features')
        st.markdown("Visualize a importância das variáveis na formação dos clusters.")

        # Calculando a importância das features como o desvio padrão das médias dos clusters
        feature_importance = df_filtered[self.binary_columns + ['cluster']].groupby('cluster').mean().std().sort_values(ascending=False)
        
        # Preparando dados para Pareto (cumulativo)
        cum_percentage = feature_importance.cumsum() / feature_importance.sum() * 100

        fig = go.Figure()

        # Adicionando barras
        fig.add_trace(go.Bar(
            x=feature_importance.index,
            y=feature_importance.values,
            name='Importância',
            marker_color='#0BAB7C'
        ))

        # Adicionando linha de porcentagem cumulativa
        fig.add_trace(go.Scatter(
            x=feature_importance.index,
            y=cum_percentage.values,
            name='Porcentagem Cumulativa',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='#145214', width=2)
        ))

        # Configurando layout com dois eixos Y
        fig.update_layout(
            title='Importância de Features por Cluster',
            xaxis=dict(title='Variável'),
            yaxis=dict(title='Desvio Padrão', side='left'),
            yaxis2=dict(title='Porcentagem Cumulativa (%)', overlaying='y', side='right'),
            legend=dict(x=0.01, y=0.99),
            barmode='group',
            template='plotly_white'
        )

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
            'Histograma de Variável Numérica': 'plot_histogram',
            'Scatter Plot 3D de Variáveis Selecionadas': 'plot_3d_scatter',
            'Gráfico de Densidade Kernel (KDE) por Cluster': 'plot_kde',
            'Gráfico de Pareto para Análise de Importância de Features': 'plot_pareto'
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

        # Boxplot para Visualização de Variáveis Numéricas
        if 'Boxplot de Variáveis Numéricas por Cluster' in selected_plots:
            self.plot_boxplot(df_filtered)

        # Scatter Plot
        if 'Scatter Plot de Variáveis Selecionadas' in selected_plots:
            self.plot_scatter(df_filtered)

        # Histograma de Variável Numérica
        if 'Histograma de Variável Numérica' in selected_plots:
            self.plot_histogram(df_filtered)


        # Pareto Plot
        if 'Gráfico de Pareto para Análise de Importância de Features' in selected_plots:
            self.plot_pareto(df_filtered)


# Função para construir a página
def build_page():
    data_path = 'KDD/ClustResult/dfKmeans.parquet'
    app = ClusterAnalysisApp(data_path)
    app.build_page()

