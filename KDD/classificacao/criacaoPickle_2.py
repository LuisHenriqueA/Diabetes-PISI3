import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
import pickle
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
    return scaler

scalers = {}
colunas_para_normalizar = ['GenHlth', 'Age', 'MentHlth', 'PhysHlth', 'Income', 'Education', 'BMI', 'Diabetes_012']
for coluna in colunas_para_normalizar:
    scalers[coluna] = normalizar_coluna(base, coluna)

# Separando features e target
X = base.drop(columns=['AnyHealthcare', 'HvyAlcoholConsump', 'Fruits', 'Sex', 'Veggies', 'CholCheck', 'DiffWalk'])
y = base['DiffWalk']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=77)

# Balanceamento dos dados de treino usando ADASYN
print("Aplicando ADASYN")
adasyn = ADASYN(random_state=77)
X_res, y_res = adasyn.fit_resample(X_train, y_train)

# Ajustando os hiperparâmetros do modelo XGBoost
print("Ajustando hiperparametros")
xgb_model = xgb.XGBClassifier(random_state=13)

# Treinando o modelo nos dados balanceados de treino com um conjunto de validação
xgb_model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=True)

# Avaliação no conjunto de teste
y_test_pred = xgb_model.predict(X_test)
print("Classification Report (Teste):")
print(classification_report(y_test, y_test_pred, target_names=['Não tem dificuldade', 'Tem dificuldade']))

# Salvando o modelo, nomes das features, e scalers com pickle
with open('xgboost_model.pkl', 'wb') as model_file:
    pickle.dump((xgb_model, X.columns.tolist(), scalers), model_file)

print("Modelo, nomes das features e scalers salvos com sucesso!")
