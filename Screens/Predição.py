import streamlit as st
import pickle
import numpy as np

class PredicaoPage:
    def __init__(self):
        # Carregar o modelo, as features e os scalers salvos
        with open('random_forest_model.pkl', 'rb') as model_file:
            self.model, self.feature_names, self.scalers = pickle.load(model_file)

    def normalizar_input(self, inputs):
        normalized_inputs = []
        for feature in self.feature_names:
            value = inputs[feature]
            scaler = self.scalers.get(feature)
            if scaler is not None:
                # Se o valor for um número, ele pode ser normalizado
                normalized_value = scaler.transform(np.array([value]).reshape(-1, 1))
                normalized_inputs.append(normalized_value[0][0])
            else:
                # Caso contrário, adicione o valor diretamente
                normalized_inputs.append(value)
        return np.array(normalized_inputs).reshape(1, -1)

    def build_page(self):
        # Título da página
        st.title("Predição de Dificuldade em Subir Escadas")

        # Mapeamento dos inputs para valores numéricos
        diabetes_map = {"não tenho": 0, "sou pré-diabético": 1, "sou diabético": 2}
        highbp_map = {"não tenho": 0, "tenho": 1}
        highchol_map = {"não tenho": 0, "tenho": 1}
        smoker_map = {"não sou": 0, "sou": 1}
        stroke_map = {"não tive": 0, "já tive": 1}
        heartdisease_map = {"não tive": 0, "já tive": 1}
        physactivity_map = {"não": 0, "sim": 1}
        nodocbccost_map = {"não": 0, "sim": 1}
        genhlth_map = {1, 2, 3, 4, 5}
        age_map = {
            "18-24 anos": 1, "25-29 anos": 2, "30-34 anos": 3,
            "35-39 anos": 4, "40-44 anos": 5, "45-49 anos": 6,
            "50-54 anos": 7, "55-59 anos": 8, "60-64 anos": 9,
            "65-69 anos": 10, "70-74 anos": 11, "75-79 anos": 12,
            "80 anos ou mais": 13
        }
        education_map = {
            "Nunca frequentou escola ou somente jardim de infância": 1,
            "1ª a 8ª série (elementary)": 2,
            "9ª a 11ª série (some high school)": 3,
            "Concluiu o ensino médio (high school graduate)": 4,
            "Faculdade (college) com 1 a 3 anos": 5,
            "Faculdade (college) com 4 anos ou mais": 6
        }
        income_map = {
            "Menos de 10,000": 1, "10,000 a 14,999": 2, "15,000 a 19,999": 3,
            "20,000 a 24,999": 4, "25,000 a 34,999": 5,
            "35,000 a 49,999": 6, "50,000 a 74,999": 7, "75,000 ou mais": 8
        }

        # Criar campos de input para cada feature necessária
        inputs = {}

        inputs['Diabetes_012'] = diabetes_map[st.selectbox(
            "Indique se você tem algum grau de diabetes",
            options=list(diabetes_map.keys())
        )]

        inputs['HighBP'] = highbp_map[st.selectbox(
            "Indique se você tem pressão alta",
            options=list(highbp_map.keys())
        )]

        inputs['HighChol'] = highchol_map[st.selectbox(
            "Indique se você tem colesterol alto",
            options=list(highchol_map.keys())
        )]

        inputs['BMI'] = st.number_input(
            "Indique seu índice de massa corporal (IMC)",
            value=0,
            step=1,
            format="%d"
        )

        inputs['Smoker'] = smoker_map[st.selectbox(
            "Indique se você é fumante",
            options=list(smoker_map.keys())
        )]

        inputs['Stroke'] = stroke_map[st.selectbox(
            "Indique se você já teve um AVC",
            options=list(stroke_map.keys())
        )]

        inputs['HeartDiseaseorAttack'] = heartdisease_map[st.selectbox(
            "Indique se você sofreu de doença coronariana ou infarto do miocárdio",
            options=list(heartdisease_map.keys())
        )]

        inputs['PhysActivity'] = physactivity_map[st.selectbox(
            "Você praticou atividade física nos últimos 30 dias?",
            options=list(physactivity_map.keys())
        )]

        inputs['NoDocbcCost'] = nodocbccost_map[st.selectbox(
            "Houve algum momento nos últimos 12 meses em que você precisou consultar um médico, mas não pôde por causa do custo?",
            options=list(nodocbccost_map.keys())
        )]

        inputs['GenHlth'] = st.selectbox(
            'Em uma escala de 1 a 5, como você classificaria sua saúde no geral, onde "1" é muito bom e "5" é muito ruim?',
            options=list(genhlth_map)
        )

        inputs['MentHlth'] = st.number_input(
            "Agora, pensando na sua saúde mental, que inclui estresse, depressão e problemas emocionais, por quantos dias durante os últimos 30 dias sua saúde mental não foi boa?",
            value=0,
            step=1,
            format="%d"
        )

        inputs['PhysHlth'] = st.number_input(
            "Agora, pensando na sua saúde física, que inclui doenças e lesões físicas, durante quantos dias durante os últimos 30 dias a sua saúde física não foi boa?",
            value=0,
            step=1,
            format="%d"
        )

        inputs['Age'] = age_map[st.selectbox(
            "Selecione a sua faixa de idade",
            options=list(age_map.keys())
        )]

        inputs['Education'] = education_map[st.selectbox(
            "Qual seu grau de escolaridade?",
            options=list(education_map.keys())
        )]

        inputs['Income'] = income_map[st.selectbox(
            "Qual sua faixa de renda média?",
            options=list(income_map.keys())
        )]

        # Quando o usuário clicar no botão "Prever"
        if st.button("Prever"):
            # Normalizar os dados de input
            normalized_inputs = self.normalizar_input(inputs)
            
            # Fazer a previsão
            prediction = self.model.predict(normalized_inputs)
            
            # Exibir o resultado
            st.write(f"A previsão é: **{prediction[0]}**")

def build_page():
    page = PredicaoPage()
    page.build_page()


