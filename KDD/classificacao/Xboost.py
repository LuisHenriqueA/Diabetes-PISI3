import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import shap


# Carregando o dataset
print("Carregando dataset")
base = pd.read_parquet('KDD/classificacao/dfCleaned.parquet')

# Normalizando colunas
def normalizar_coluna(df, nome_coluna):
    scaler = MinMaxScaler()
    df[nome_coluna] = scaler.fit_transform(df[[nome_coluna]])

colunas_para_normalizar = ['GenHlth', 'Age', 'MentHlth', 'PhysHlth', 'Income', 'Education', 'BMI', 'Diabetes_012']
for coluna in colunas_para_normalizar:
    normalizar_coluna(base, coluna)

# Separando features e target
X = base.drop(columns=['AnyHealthcare', 'HvyAlcoholConsump', 'Fruits', 'Sex', 'Veggies', 'CholCheck', 'DiffWalk'])
y = base['DiffWalk']

# Ajustando os hiperparâmetros do modelo XGBoost
print("Ajustando hiperparametros")
xgb_model = xgb.XGBClassifier(
    random_state=13,
)
# Configurando a validação cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

# Realizando predições com validação cruzada
y_pred = cross_val_predict(xgb_model, X, y, cv=cv)

# Relatório de classificação após a validação cruzada
print("Classification Report (Validação Cruzada):")
print(classification_report(y, y_pred, target_names=['Não tem dificuldade', 'Tem dificuldade']))

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=77)
'''
'''
print("Aplicando ADASYN")

# Balanceamento dos dados de treino usando ADASYN
adasyn = ADASYN(random_state=77)
X_res, y_res = adasyn.fit_resample(X_train, y_train)

# Treinando o modelo nos dados balanceados de treino com um conjunto de validação
xgb_model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=True)

# Avaliação no conjunto de treino
y_train_pred = xgb_model.predict(X_res)
print("Classification Report (Treino):")
print(classification_report(y_res, y_train_pred, target_names=['Não tem dificuldade', 'Tem dificuldade']))

# Avaliação no conjunto de teste
y_test_pred = xgb_model.predict(X_test)
print("Classification Report (Teste):")
print(classification_report(y_test, y_test_pred, target_names=['Não tem dificuldade', 'Tem dificuldade']))

# Criando um dicionário para as traduções
traducoes = {
    "Age": "Idade",
    "GenHlth": "Saúde Geral",
    "PhysHlth": "Saúde Física",
    "BMI": "Índice de Massa Corporal",
    "Income": "Renda",
    "PhysActivity": "Atividade Física",
    "Education": "Educação",
    "MentHlth": "Saúde Mental",
    "HighBP": "Pressão Alta",
    "Smoker": "Fumante",
    "Stroke": "AVC",
    "NoDocbcCost": "Sem Consulta Médica por Custo",
    "HeartDiseaseorAttack": "Doença Cardíaca ou Ataque",
    "Diabetes_012": "Diabetes",
    "HighChol": "Colesterol Alto"
}

# Renomeando as colunas de X_test
X_test_traduzido = X_test.rename(columns=traducoes)

# Ajuste do modelo (se necessário)
xgb_model.fit(X_res, y_res)

# Criando o objeto explainer do SHAP
explainer = shap.TreeExplainer(xgb_model)

# Calculando os valores SHAP para o conjunto de teste traduzido
shap_values = explainer.shap_values(X_test_traduzido)

# Sumário plot para importância global das features traduzidas
shap.summary_plot(shap_values, X_test_traduzido)

# Para importância média das features traduzidas
shap.summary_plot(shap_values, X_test_traduzido, plot_type="bar")
