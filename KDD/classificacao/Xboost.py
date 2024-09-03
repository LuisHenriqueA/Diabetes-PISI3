import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando o dataset
print("Carregando dataset")
base = pd.read_parquet('/Users/alanalins/pisi3/Diabetes-PISI3/KDD/classificacao/dfCleaned.parquet')

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
    random_state=2,
    max_depth=3,               
    min_child_weight=6,        
    gamma=0.6,                 
    subsample=0.6,             
    colsample_bytree=0.6,      
    learning_rate=0.01,        
    n_estimators=1000,         
    reg_alpha=0.2,             
    reg_lambda=0.3,            
    scale_pos_weight=1,        
    early_stopping_rounds=50,  
    eval_metric="logloss"
)

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

#
