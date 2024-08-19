import streamlit as st
from utils.header import add_header

def build_page():
    st.markdown('''
        <style>
        .justificado {
            text-align: justify;
        }
        </style>
        <div class= "justificado">
            <h1>Aprendizado de máquina aplicado no contexto de Diabetes</h1>
            <br>
            Grupo: Alana lins; David Erick; Davi Vieira; Luis Henrique;
            </br>
            <br>
            O presente estudo aborda a Diabetes Mellitus (DM) como um distúrbio metabólico que afeta milhões de pessoas globalmente, que destaca-se pela sua complexidade e impacto na qualidade de vida dos pacientes.
            Com o aumento da prevalência da doença, surge a importância de estudos para entender o impacto da Diabetes na vida dos pacientes. 
            Nesse contexto, a aplicação de técnicas de aprendizado de máquina, algoritmos de clusterização e classificação, 
            surge como uma ferramenta  para analisar dados e identificar padrões que indicam impactos causados nos pacientes diabéticos e pré-diabéticos por conta da doença,
            essa pesquisa visa utilizar as técnicas de aprendizado de máquina na base de dados Diabetes Health Indicator Dataset para conduzir sua análise e investigação. 
            <br>
        </div>
    ''', unsafe_allow_html=True)

def render_page(page_name):
    if page_name == "Home🏠":
        build_page()
    elif page_name == "Dashboards📈":
        import Screens.Dashboards
        Screens.Dashboards.build_page()
    elif page_name == "Clusterização🔍":
        import Screens.Clusterização
        Screens.Clusterização.build_page()
    elif page_name == "Classificação🔍":
        import Screens.Classificacao
        Screens.Classificacao.build_page()
        
# Opções de navegação
def configure_sidebar():
    st.sidebar.markdown(
        """
        <style>
        .stRadio *{
            color: #FBFAF3;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown('''
        <style>
        .sidebar_h2{
            color: #FBFAF3;
            }
        .stRadio *{
            font-size: 17px;
            }              
        </style>
        <h2 class="sidebar_h2">Escolha uma página</h2>
                        
    ''', unsafe_allow_html=True)
    
    page = st.sidebar.radio("Páginas", ["Home🏠", "Dashboards📈", "Clusterização🔍", "Classificação🔍" ], key='sidebar',  label_visibility="hidden")
    return page

def main():
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
    add_header()
    page = configure_sidebar()
    render_page(page)

main()