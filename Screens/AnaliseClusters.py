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
        self.color_map = {0: '#0BAB7C', 1: '#A6DFA3'}  # Novas cores para os clusters

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

 
    def plot_treemap(self, df_filtered):
        """
        Cria um treemap para mostrar a distribuição dos clusters na coluna 'Não, pré ou Diabético'.
        """
        st.subheader('Treemap da Distribuição dos Clusters na Coluna "Não, pré ou Diabético"')
        st.markdown("Este treemap mostra a diferença entre os clusters para a coluna 'Não, pré ou Diabético'.")

        # Substituir os valores na coluna 'Não, pré ou Diabético'
        df_filtered['Não, pré ou Diabético'] = df_filtered['Não, pré ou Diabético'].replace({
            0.0: 'Não diabéticos',
            1.0: 'Pré diabéticos',
            2.0: 'Diabéticos'
        })

        # Contar o número de indivíduos em cada combinação de 'cluster' e 'Não, pré ou Diabético'
        df_counts = df_filtered.groupby(['cluster', 'Não, pré ou Diabético']).size().reset_index(name='Count')

        fig = px.treemap(
            df_counts,
            path=['cluster', 'Não, pré ou Diabético'],
            values='Count',  # Usar a coluna 'Count' para definir o tamanho das caixas
            color='cluster',
            color_discrete_map=self.color_map,
            color_continuous_scale='Greens',
            color_discrete_sequence=['#0BAB7C', '#A6DFA3'],
        )

        fig.update_traces(textinfo="label+value")
        st.plotly_chart(fig)

    def plot_bar_charts(self, df_filtered):
        """
        Cria gráficos de barras para variáveis binárias por cluster com a opção de seleção de variável.
        """
        labels_map = {
            'Sexo': {0: 'Feminino', 1: 'Masculino'},
            'Fumante': {0: 'Não fumante', 1: 'Fumante'},
            'Consumo excessivo de álcool': {0: 'Não, consumo não excessivo', 1: 'Sim, consumo excessivo'},
            'Pratica atividade física': {0: 'Não pratica', 1: 'Pratica'},
            'Consome frutas diariamente': {0: 'Não consome', 1: 'Consome'},
            'Consome vegetais diariamente': {0: 'Não consome', 1: 'Consome'},
            'Pressão alta': {0: 'Não', 1: 'Sim'},
            'Colesterol alto': {0: 'Não', 1: 'Sim'},
            'AVC': {0: 'Não teve', 1: 'Já teve'},
            'Doença Coronariana ou Infarto do Miocárdio': {0: 'Não teve', 1: 'Já teve'},
            'Dificuldade em Subir Escadas': {0: 'Não tem dificuldade', 1: 'Tem dificuldade'}
        }

        # Filtro de variáveis binárias
        st.subheader('Distribuição das Variáveis Binarias por Cluster')
        st.markdown("Gráfico de barras mostrando a distribuição de valores binários por cluster.")
        selected_var = st.selectbox(
            'Selecione a Variável Binária para Visualizar',
            options=self.binary_columns,
            index=0  # Define o valor padrão para a primeira opção
        )

        if selected_var:
            st.write(f'Distribuição para {selected_var}')

            df_filtered[selected_var + '_label'] = df_filtered[selected_var].map(labels_map[selected_var])

            fig = px.histogram(
                df_filtered, 
                x=selected_var + '_label', 
                color='cluster', 
                barmode='group', 
                title=f'Distribuição de {selected_var} por Cluster',
                color_discrete_map=self.color_map,
                color_discrete_sequence=['#0BAB7C', '#A6DFA3']
            )
            fig.update_traces(marker=dict(line=dict(width=0)))  # Remove linhas em volta das barras para destacar as cores
            st.plotly_chart(fig)


    # Outras funções permanecem as mesmas
    def plot_heatmap(self, df_filtered):
        """
        Cria um heatmap interativo de frequência das variáveis binárias por cluster,
        mostrando os valores diretamente nas células.
        """
        st.subheader('Heatmap de Frequência das Variáveis Binárias')
        st.markdown("Este heatmap mostra a frequência média de cada variável binária dentro de cada cluster.")

        # Filtrando as colunas binárias e a coluna de cluster
        heatmap_data = df_filtered[self.binary_columns + ['cluster']]
    
        # Garantindo que a coluna 'cluster' seja do tipo int
        heatmap_data['cluster'] = heatmap_data['cluster'].astype(int)
    
        # Agrupando por cluster e calculando a média
        heatmap_data = heatmap_data.groupby('cluster').mean().T

        # Criando o heatmap interativo usando Plotly
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Cluster", y="Variável", color="Frequência Média"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='greens',
            text_auto=True,
            aspect="auto",  # Ajusta automaticamente a proporção dos quadrados
            width=700,     # Aumenta a largura total do gráfico
            height=500
        )
    
        # Atualizando os rótulos do eixo X para mostrar apenas 0 e 1
        fig.update_xaxes(type='category', tickvals=[0, 1], ticktext=['0', '1'])
    
        # Ajustando o espaçamento dos quadrados para torná-los maiores
        fig.update_xaxes(tick0=0, dtick=1)
        fig.update_yaxes(tick0=0, dtick=1)
    
        # Atualizando a formatação dos valores nas células
        fig.update_traces(texttemplate="%{z:.2f}", textfont_size=12)
    
        # Exibindo o gráfico no Streamlit
        st.plotly_chart(fig)




    def plot_radar_chart(self, df_filtered, cluster):
        """
        Cria um gráfico de radar interativo otimizado para múltiplas variáveis binárias de um cluster.
        """
        st.subheader('Gráfico de Radar das Variáveis Binárias')
        st.markdown("Este gráfico mostra a frequência média de cada variável binária em forma de um radar para facilitar a compreensão.")

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
                         title=f'Boxplot de IMC por Cluster', color_discrete_map=self.color_map, color_discrete_sequence=['#0BAB7C', '#A6DFA3'])
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
                               opacity=0.7, color_discrete_sequence=['#0BAB7C', '#A6DFA3'])
            st.plotly_chart(fig)


    def plot_pareto(self, df_filtered):
        """
        Cria um gráfico de Pareto para mostrar a importância das variáveis binárias na formação dos clusters.
        """
        st.subheader('Gráfico de Pareto para Análise de Importância de Features binárias')
        st.markdown("Visualize a importância das variáveis binárias na formação dos clusters.")

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
        

        # Filtro para seleção de gráficos
        st.subheader('Seleção de Gráficos')
        available_plots = {
            'Análise da condição diabética em cada cluster': 'plot_treemap',
            'Distribuição das Variáveis Binárias por Cluster': 'plot_bar_charts',
            'Heatmap de Frequência': 'plot_heatmap',
            'Gráfico Radar para Análise de Múltiplas Variáveis Binárias': 'plot_radar_chart',
            'Boxplot de Variáveis Numéricas por Cluster': 'plot_boxplot',
            'Histograma de Variável Numérica': 'plot_histogram',
            'Gráfico de Pareto para Análise de Importância de Features': 'plot_pareto'
        }
        selected_plots = st.multiselect('Selecione os Gráficos que Deseja Visualizar', list(available_plots.keys()), default=list(available_plots.keys()))

        if 'Análise da condição diabética em cada cluster' in selected_plots:
            self.plot_treemap(df_filtered)

        # Gráficos Interativos para Colunas Binárias
        if 'Distribuição das Variáveis Binárias por Cluster' in selected_plots:
            self.plot_bar_charts(df_filtered)

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
