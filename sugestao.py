def build_dot_plot(df: pd.DataFrame):
    st.markdown('<h3>Gráfico de Pontos: Saúde Geral por Condição Diabética</h3>', unsafe_allow_html=True)

    condition_map = {0: 'Não diabético', 1: 'Pré diabético', 2: 'Diabético'}
    saúde_map = {1: 'Excelente', 2: 'Muito boa', 3: 'Boa', 4: 'Regular', 5: 'Ruim'}
    df['Condição diabética'] = df['Não, pré ou Diabético'].map(condition_map)
    df['Saúde geral'] = df['Saúde geral'].map(saúde_map)
    
    count_df = df.groupby(['Condição diabética', 'Saúde geral']).size().reset_index(name='Contagem')

    fig = px.scatter(
        count_df,
        x='Contagem',
        y='Saúde geral',
        color='Condição diabética',
        size='Contagem',
        hover_data={'Contagem': True, 'Condição diabética': True},
        labels={'Contagem': 'Quantidade', 'Saúde geral': 'Saúde Geral'},
        title='Distribuição da Saúde Geral por Condição Diabética'
    )

    fig.update_layout(
        yaxis=dict(categoryorder='array', categoryarray=['Excelente', 'Muito boa', 'Boa', 'Regular', 'Ruim']),
        xaxis_title='Quantidade',
        yaxis_title='Saúde Geral'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def build_interactive_scatter_plot(df: pd.DataFrame):
    st.markdown('<h3>Gráfico de Dispersão Interativo: Saúde Geral por Condição Diabética</h3>', unsafe_allow_html=True)

    condition_map = {0: 'Não diabético', 1: 'Pré diabético', 2: 'Diabético'}
    saúde_map = {1: 'Excelente', 2: 'Muito boa', 3: 'Boa', 4: 'Regular', 5: 'Ruim'}
    df['Condição diabética'] = df['Não, pré ou Diabético'].map(condition_map)
    df['Saúde geral'] = df['Saúde geral'].map(saúde_map)
    
    diabetes_options = ['Todos'] + sorted(df['Condição diabética'].unique())
    selected_option = st.selectbox('Selecione a condição diabética para visualizar:', diabetes_options)

    if selected_option != 'Todos':
        df = df[df['Condição diabética'] == selected_option]
    
    count_df = df.groupby(['Condição diabética', 'Saúde geral']).size().reset_index(name='Contagem')

    fig = px.scatter(
        count_df,
        x='Saúde geral',
        y='Contagem',
        color='Condição diabética',
        size='Contagem',
        hover_data={'Contagem': True, 'Condição diabética': True},
        labels={'Contagem': 'Quantidade', 'Saúde geral': 'Saúde Geral'},
        title='Distribuição Interativa da Saúde Geral por Condição Diabética'
    )

    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=['Excelente', 'Muito boa', 'Boa', 'Regular', 'Ruim']),
        yaxis_title='Quantidade',
        xaxis_title='Saúde Geral'
    )

    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    
    st.plotly_chart(fig, use_container_width=True)
