import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import ADASYN

# Carregando o dataset
base = pd.read_parquet('KDD/classificacao/dfCleaned.parquet')

# Função para normalizar colunas
def normalizar_coluna(df, nome_coluna):
    scaler = MinMaxScaler()
    df[nome_coluna] = scaler.fit_transform(df[[nome_coluna]])

# Normalizando colunas
colunas_para_normalizar = ['GenHlth', 'Age', 'MentHlth', 'PhysHlth', 'Income', 'Education', 'BMI', 'Diabetes_012']
for coluna in colunas_para_normalizar:
    normalizar_coluna(base, coluna)

# Separando features e target
X = base.drop(columns=['AnyHealthcare', 'HvyAlcoholConsump', 'Fruits', 'Sex', 'Veggies', 'CholCheck', 'DiffWalk'])
y = base['DiffWalk']

# Renomeando as classes
y = y.replace({0: 'Não tem dificuldade', 1: 'Tem dificuldade'})

# Definindo o modelo ajustado para reduzir overfitting
rf = RandomForestClassifier(
    random_state=77,
    n_estimators=100,          # Mantendo o número de árvores moderado
    max_depth=7,               # Limitar a profundidade das árvores
    min_samples_split=10,      # Aumentar o número mínimo de amostras para dividir um nó
    min_samples_leaf=5,        # Aumentar o número mínimo de amostras por folha
    max_features='sqrt'        # Limitar o número de features consideradas para cada split
) 
# Validação cruzada e obtenção de previsões
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
y_pred = cross_val_predict(rf, X, y, cv=cv)

# Relatório de classificação após a validação cruzada
print("Classification Report (Validação Cruzada):")
print(classification_report(y, y_pred, target_names=['Não tem dificuldade', 'Tem dificuldade']))

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=77)

# Balanceamento dos dados de treino usando SMOTEENN
adasyn = ADASYN(random_state=77)
X_res, y_res = adasyn.fit_resample(X_train, y_train)



# Treinando o modelo nos dados balanceados de treino
rf.fit(X_res, y_res)

# Avaliação no conjunto de treino
y_train_pred = rf.predict(X_res)
print("Classification Report (Treino):")
print(classification_report(y_res, y_train_pred, target_names=['Não tem dificuldade', 'Tem dificuldade']))

# Avaliação no conjunto de teste
y_test_pred = rf.predict(X_test)
print("Classification Report (Teste):")
print(classification_report(y_test, y_test_pred, target_names=['Não tem dificuldade', 'Tem dificuldade']))