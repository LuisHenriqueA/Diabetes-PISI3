import streamlit as st

def add_header():
    c1, c2 = st.columns([.05,.95])
    with c1:
        st.image("Assets/Logo.png", width=80)
    with c2:
        st.markdown(
            '''
        <h2>Dattageters</h2>
        ''', unsafe_allow_html=True)