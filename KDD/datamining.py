import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

caminho_arquivo = 'dfCleaned.csv'
df = pd.read_csv(caminho_arquivo, encoding='utf-8', low_memory=False)
print(df.head())
selected_columns = [
    'NoDocbcCost', 
    'AnyHealthcare', 
    'Veggies', 
    'Fruits', 
    'PhysActivity', 
    'HeartDiseaseorAttack', 
    'Smoker', 
    'CholCheck', 
    'HighBP', 
    'Diabetes_012', 
]
X = df[selected_columns]
# Ver as colunas do DataFrame
print(df.columns)
y = df['DiffWalk']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar as features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar e treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test_scaled)

# Avaliar o modelo
classification_rep = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Exibir relatório de classificação e acurácia
print("\nRelatório de classificação:")
print(classification_rep)

