import streamlit as st

st.set_page_config(
    page_title = "PISI 3 - Diabetes",
    layout = "wide",
    menu_items = {
        'About': '''Sistema desenvolvido para cadeira de PISI 3
        Tema: Diabetes
        Autores: Alana Lins, Davi Vieira, David Erick, Luis Henrique
        Dataset link: www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
        '''
    }
)

st.markdown(f'''
    <h1>Aprendizado de máquina aplicado no contexto de Diabetes</h1>
    <br>
    O presente estudo aborda a Diabetes Mellitus (DM) como um distúrbio metabólico que afeta milhões de pessoas globalmente, que destaca-se pela sua complexidade e impacto na qualidade de vida dos pacientes.
    Com o aumento da prevalência da doença, surge a importância de estudos para entender o impacto da Diabetes na vida dos pacientes. 
    Nesse contexto, a aplicação de técnicas de aprendizado de máquina, algoritmos de clusterização e classificação, 
    surge como uma ferramenta  para analisar dados e identificar padrões que indicam impactos causados nos pacientes diabéticos e pré-diabéticos por conta da doença,
    essa pesquisa visa utilizar as técnicas de aprendizado de máquina na base de dados Diabetes Health Indicator Dataset para conduzir sua análise e investigação. 
    <br>
''', unsafe_allow_html=True)