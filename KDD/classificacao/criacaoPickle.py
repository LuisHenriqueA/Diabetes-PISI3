import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report  # Import correto
import pickle


# Carregando o dataset
base = pd.read_parquet('KDD/dfCleaned.parquet')

def normalizar_coluna(df, nome_coluna):
    scaler = MinMaxScaler()
    coluna = df[[nome_coluna]]
    df[nome_coluna] = scaler.fit_transform(coluna)
    return scaler

# Normalizando colunas
scalers = {}
colunas_para_normalizar = ['GenHlth', 'Age', 'MentHlth', 'PhysHlth', 'Income', 'Education', 'BMI', 'Diabetes_012']
for coluna in colunas_para_normalizar:
    scalers[coluna] = normalizar_coluna(base, coluna)

# Incluindo a feature `NoDocbcCost`
X = base.drop(columns=['AnyHealthcare', 'HvyAlcoholConsump', 'Fruits', 'Sex', 'Veggies', 'CholCheck', 'DiffWalk'])
X['NoDocbcCost'] = base['NoDocbcCost']
y = base['DiffWalk']

# Fazendo a separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Renomeando para ficar mais entendível
y_train = y_train.replace({0: 'Não tem dificuldade', 1: 'Tem dificuldade'})
y_test = y_test.replace({0: 'Não tem dificuldade', 1: 'Tem dificuldade'})

# Verificando a distribuição das classes após a separação
print("Distribuição das classes no treinamento:", y_train.value_counts())

# Aplicando RandomUnderSampler para balancear as classes
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

# Verificando a distribuição das classes após o undersampling
print("Distribuição das classes após o undersampling:", y_res.value_counts())

# Ajustando o modelo Random Forest
rf = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=9,
    min_samples_split=25,
    min_samples_leaf=15,
    max_features='sqrt',
    class_weight={'Não tem dificuldade': 1, 'Tem dificuldade': 1.010}
)

# Treinando o modelo
rf.fit(X_res, y_res)

# Avaliação do modelo para verificar se ele não está tendencioso
predictions = rf.predict(X_test)
print("Acurácia no conjunto de teste:", accuracy_score(y_test, predictions))
print("Relatório de classificação:\n", classification_report(y_test, predictions))

# Salvando o modelo, nomes das features, e scalers com pickle
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump((rf, X.columns.tolist(), scalers), model_file)

print("Modelo, nomes das features e scalers salvos com sucesso!")
