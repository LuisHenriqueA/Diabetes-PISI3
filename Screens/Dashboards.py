import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import io
from PIL import Image
from utils.plot_utils import read_df, build_dataframe
import seaborn as sns
import numpy as np
from utils.plot_utils import __transform_data

def build_page():
    build_header()
    build_body()

def build_header():
    text = '<h1>Gráficos com base de diabetes</h1>' + \
           '<p>Esta página apresenta informações a partir da seguinte base de dados importada do ' + \
           '<a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset">Kaggle</a>¹ .</p>'
    st.write(text, unsafe_allow_html=True)

def build_body():
    df = read_df('KDD/dfCleaned')
    df = __transform_data(df)
    df_bruto = read_df('DiabetesDataSet/diabetes_012_health_indicators_BRFSS2015')
    df_bruto = __transform_data(df_bruto)
    
    st.header("Escolha os Gráficos")
    
    # Dividindo a tela em duas colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Seletor de gráfico
        graph_options = {
            'Histograma Diabético': 'diabetesplot',
            'Boxplot': 'boxplot',
            'Gráfico de Venn': 'venn',
            'Gráfico de Dispersão': 'scatterplot',
            'Gráfico de Pontos': 'dotplot'
        }
        selected_graphs = st.multiselect(
            'Selecione os gráficos para mostrar', 
            options=list(graph_options.keys()), 
            default=list(graph_options.keys())
        )
    
    with col2:
        # Filtro por gênero
        gender_filter = st.selectbox(
            'Gênero',
            options=['Ambos', 'Masculino', 'Feminino'],
            format_func=lambda x: {'Ambos': 'Ambos', 'Masculino': 'Homem', 'Feminino': 'Mulher'}[x]
        )
    
    if gender_filter != 'Ambos':
        gender_value = 1 if gender_filter == 'Masculino' else 0
        df = df[df['Sexo'] == gender_value]
    
    # Chama as funções com base nas seleções
    if 'Histograma Diabético' in selected_graphs:
        build_diabetesplot_section(df)
    
    if 'Boxplot' in selected_graphs:
        build_boxplot_section(df)
    
    if 'Gráfico de Venn' in selected_graphs:
        build_venn_plot(df)
    
    if 'Gráfico de Dispersão' in selected_graphs:
        build_interactive_scatter_plot(df)
    
    if 'Gráfico de Pontos' in selected_graphs:
        build_dot_plot(df)

def build_diabetesplot_section(df: pd.DataFrame):
    st.markdown('<h3>Gráfico de barras empilhadas sobre condição diabética</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([.3, .7])
    cols = ['Fumante', 'Cobertura de saúde', 'Pratica atividade física']
    selec_col = c1.selectbox('Seleção', options=cols, key='selec_1')
    fig = create_histograma_stacked(df, selec_col)
    fig.update_layout(
        title=f'Gráfico de {selec_col} por condição diabética.',
        legend_title_text=selec_col,
        xaxis_title_text='Condição',
        yaxis_title_text='Quantidade'
    )
    c2.plotly_chart(fig, use_container_width=True)

def create_histograma_stacked(df: pd.DataFrame, series_col: str) -> go.Figure:
    # Criar mapeamentos para as colunas binárias e a coluna de condição diabética
    bin_map = {0: 'Não', 1: 'Sim'}
    condition_map = { 0: 'Não diabético', 1: 'Pré diabético', 2: 'Diabético'}
    
    # Substituir valores das colunas binárias e da condição diabética pelos textos
    df[series_col + ' Textual'] = df[series_col].map(bin_map)
    df['Condição Textual'] = df['Não, pré ou Diabético'].map(condition_map)
    
    # Filtrar valores não nulos
    df = df.query(f'`{series_col}`.notna()').copy()
    
    # Criar o histograma
    color_map = {'Sim': '#0BAB7C', 'Não': '#C7F4C2'}
    fig = px.histogram(
        df,
        x='Condição Textual',
        nbins=20,
        color=series_col + ' Textual',
        opacity=.75,
        color_discrete_map=color_map,
    )
    
    return fig

def build_boxplot_section(df: pd.DataFrame):
    st.markdown('<h3>Diagramas de Caixa</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([.3, .7])
    cols = ['IMC', 'Saúde física', 'Saúde Mental']
    selec_col = c1.selectbox('Seleção', options=cols, key='selec_2')
    reverse = c1.checkbox('Inverter orientação', value=False)
    
    # Mapear valores da condição diabética
    condition_map = {0: 'Não diabético', 1: 'Pré diabético', 2: 'Diabético'}
    df['Condição diabética'] = df['Não, pré ou Diabético'].map(condition_map)
    
    # Preparar colunas para o boxplot
    col_selected = [selec_col, 'Condição diabética']
    orientation = 'h'  # Orientação vertical por padrão
    
    if reverse:
        col_selected.reverse()
        orientation = 'v'  # Orientação horizontal se invertido

    df_plot = df[col_selected]
    
    # Criar o boxplot
    fig = px.box(df_plot, x=col_selected[0], y=col_selected[1], orientation=orientation)
    
    # Manter a cor padrão do boxplot
    fig.update_traces(marker_color='#0BAB7C')  # Cor padrão, se desejado
    fig.update_layout(
        xaxis_title=col_selected[0],
        yaxis_title=col_selected[1]
    )
    
    c2.plotly_chart(fig, use_container_width=True)

def build_venn_plot(df: pd.DataFrame):
    st.markdown('<h3>Gráficos de Venn</h3>', unsafe_allow_html=True)
    binary_cols = [
        'Pressão alta', 'Colesterol alto', 'Fumante', 
        'AVC', 'Doença Coronariana ou Infarto do Miocárdio', 'Pratica atividade física', 
        'Consome frutas diariamente', 'Consome vegetais diariamente', 'Dificuldade em se locomover'
    ]
    
    # Criar três colunas: uma para os seletores e duas para o gráfico
    col1, col2, col3 = st.columns([.3, .2, .5])  # Ajustar as larguras conforme necessário
    
    with col1:
        selected_col1 = st.selectbox("Escolha a primeira coluna", options=binary_cols, key='col1')
        # Atualizar a lista de opções para a segunda seleção, excluindo a coluna já escolhida
        remaining_cols = [col for col in binary_cols if col != selected_col1]
        selected_col2 = st.selectbox("Escolha a segunda coluna", options=remaining_cols, key='col2')
    
    with col3:
        if selected_col1 and selected_col2:
            # Contar as combinações de valores das duas colunas binárias
            counts = pd.crosstab(df[selected_col1], df[selected_col2])
            venn_counts = {
                '10': counts.loc[1, 0] if 1 in counts.index and 0 in counts.columns else 0,
                '01': counts.loc[0, 1] if 0 in counts.index and 1 in counts.columns else 0,
                '11': counts.loc[1, 1] if 1 in counts.index and 1 in counts.columns else 0
            }
            
            # Criar o gráfico de Venn
            fig, ax = plt.subplots(figsize=(8, 6))
            venn = venn2(subsets=venn_counts, set_labels=[selected_col1, selected_col2], set_colors=('#0BAB7C', '#C7F4C2'))
            for patch in venn.patches:
                if patch is not None:
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)
                
            plt.title(f'Gráfico de Venn para {selected_col1} e {selected_col2}')
            
            # Salvar o gráfico em um buffer de memória com fundo transparente
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight')
            buf.seek(0)
            
            # Converter o buffer em uma imagem PIL
            img = Image.open(buf)
            
            # Exibir o gráfico como uma imagem
            st.image(img, caption=f'Gráfico de Venn para {selected_col1} e {selected_col2}')
    
def build_interactive_scatter_plot(df: pd.DataFrame):
    st.markdown('<h3>Gráfico de Dispersão Interativo: Saúde Geral por Condição Diabética</h3>', unsafe_allow_html=True)
    dfScatter = df.copy()
    condition_map = {0: 'Não diabético', 1: 'Pré diabético', 2: 'Diabético'}
    saúde_map = {1: 'Excelente', 2: 'Muito boa', 3: 'Boa', 4: 'Regular', 5: 'Ruim'}
    dfScatter['Condição diabética'] = dfScatter['Não, pré ou Diabético'].map(condition_map)
    dfScatter['Saúde geral'] = dfScatter['Saúde geral'].map(saúde_map)
    
    # Adicionar a selectbox para o usuário selecionar os grupos de diabetes
    diabetes_options = ['Todos'] + sorted(dfScatter['Condição diabética'].unique())
    selected_option = st.selectbox('Selecione a condição diabética para visualizar:', diabetes_options, key='diabetes_options')

    # Filtrar o dataframe com base na escolha do usuário
    if selected_option != 'Todos':
        dfScatter = dfScatter[dfScatter['Condição diabética'] == selected_option]
    
    # Contar as ocorrências de "Saúde geral" por "Condição diabética"
    count_df = dfScatter.groupby(['Condição diabética', 'Saúde geral']).size().reset_index(name='Contagem')

    # Criar o gráfico de dispersão interativo
    fig = px.scatter(
        count_df,
        x='Saúde geral',
        y='Contagem',
        color='Condição diabética',
        size='Contagem',  # O tamanho dos pontos reflete a contagem
        hover_data={'Contagem': True, 'Condição diabética': True},
        labels={'Contagem': 'Quantidade', 'Saúde geral': 'Saúde Geral'},
        title='Distribuição Interativa da Saúde Geral por Condição Diabética',
        color_discrete_map={'Não diabético': '#0BAB7C', 'Pré diabético': '#C7F4C2', 'Diabético': '#064E3B'}
    )

    # Ajustar o layout para uma visualização mais clara
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=['Excelente', 'Muito boa', 'Boa', 'Regular', 'Ruim']),
        yaxis_title='Quantidade',
        xaxis_title='Saúde Geral'
    )

    # Adicionar controles interativos
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    
    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

def build_dot_plot(df: pd.DataFrame):
    st.markdown('<h3>Gráfico de Pontos: Saúde Geral por Condição Diabética</h3>', unsafe_allow_html=True)

    condition_map = {0: 'Não diabético', 1: 'Pré diabético', 2: 'Diabético'}
    saúde_map = {1: 'Excelente', 2: 'Muito boa', 3: 'Boa', 4: 'Regular', 5: 'Ruim'}
    df['Condição diabética'] = df['Não, pré ou Diabético'].map(condition_map)
    df['Saúde geral'] = df['Saúde geral'].map(saúde_map)

    diabetes_options2 = ['Todos'] + sorted(df['Condição diabética'].unique())
    selected_option2 = st.selectbox('Selecione a condição diabética para visualizar:', diabetes_options2, key='diabetes_options2')

    # Filtrar o dataframe com base na escolha do usuário
    if selected_option2 != 'Todos':
        df = df[df['Condição diabética'] == selected_option2]
    
    count_df = df.groupby(['Condição diabética', 'Saúde geral']).size().reset_index(name='Contagem')

    fig = px.scatter(
        count_df,
        x='Contagem',
        y='Saúde geral',
        color='Condição diabética',
        size='Contagem',
        hover_data={'Contagem': True, 'Condição diabética': True},
        labels={'Contagem': 'Quantidade', 'Saúde geral': 'Saúde Geral'},
        title='Distribuição da Saúde Geral por Condição Diabética',
        color_discrete_map={'Não diabético': '#0BAB7C', 'Pré diabético': '#C7F4C2', 'Diabético': '#064E3B'}
    )

    fig.update_layout(
        yaxis=dict(categoryorder='array', categoryarray=['Excelente', 'Muito boa', 'Boa', 'Regular', 'Ruim']),
        xaxis_title='Quantidade',
        yaxis_title='Saúde Geral'
    )
    
    st.plotly_chart(fig, use_container_width=True)