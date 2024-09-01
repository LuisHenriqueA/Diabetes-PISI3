import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np

# Carregando o dataset
base = pd.read_parquet('KDD/classificacao/dfCleaned.parquet')

def normalizar_coluna(df, nome_coluna):
    # Inicializar o MinMaxScaler
    scaler = MinMaxScaler()
    
    # Selecionar a coluna especificada
    coluna = df[[nome_coluna]]
    
    # Normalizar a coluna
    df[nome_coluna] = scaler.fit_transform(coluna)

# Normalizando colunas
colunas_para_normalizar = ['GenHlth', 'Age', 'MentHlth', 'PhysHlth', 'Income', 'Education', 'BMI', 'Diabetes_012']
for coluna in colunas_para_normalizar:
    normalizar_coluna(base, coluna)

# Dropando colunas com menor correlação e a target
X = base.drop(columns=['AnyHealthcare', 'HvyAlcoholConsump', 'Fruits', 'Sex', 'Veggies', 'CholCheck', 'DiffWalk'])
y = base['DiffWalk']

# Fazendo a separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Renomeando para ficar mais entendível
y_train = y_train.replace({0: 'Não tem dificuldade', 1: 'Tem dificuldade'})
y_test = y_test.replace({0: 'Não tem dificuldade', 1: 'Tem dificuldade'})

# Definindo o RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
# Aplicando para X e y
X_res, y_res = rus.fit_resample(X_train, y_train)

# Ajustando o modelo Random Forest com pesos de classe modificados
rf = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=9,
    min_samples_split=25,
    min_samples_leaf=15,
    max_features='sqrt',
    class_weight={'Não tem dificuldade': 1, 'Tem dificuldade': 1.010}
)

# Criando o objeto de validação cruzada com 5 folds estratificados
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Usando cross_val_predict para obter previsões para cada fold (dados de treino)
y_pred_train = cross_val_predict(rf, X_res, y_res, cv=cv)

# Treinando o modelo nos dados de treino completos
rf.fit(X_res, y_res)

# Fazendo previsões nos dados de teste
y_pred_test = rf.predict(X_test)

# Calculando e exibindo o relatório de classificação para dados de treino
accuracy_train = accuracy_score(y_res, y_pred_train)
report_train = classification_report(y_res, y_pred_train)

# Calculando e exibindo o relatório de classificação para dados de teste
accuracy_test = accuracy_score(y_test, y_pred_test)
report_test = classification_report(y_test, y_pred_test)

print(f"Acurácia nos dados de treino Random forest com peso maior para o que tem dificuldade: {accuracy_train:.2f}")
print(report_train)

print(f"Acurácia nos dados de teste Random forest com peso maior para o que tem dificuldade: {accuracy_test:.2f}")
print(report_test)

print("Resultados no Treino com StratifiedKFold:")
print(f"Acurácia no treino: {accuracy_train:.2f}")
print(report_train)

print("\nResultados no Teste:")
print(f"Acurácia no teste: {accuracy_test:.2f}")
print(report_test)

# Comparando diretamente acurácia ou outras métricas, se necessário
if abs(accuracy_train - accuracy_test) < 0.05:  # Tolerância de 5%
    print("As métricas de treino e teste são consistentes.")
else:
    print("Há uma diferença significativa entre treino e teste.")


# Calculando a importância das features
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Exibindo a importância das features
print("Importância das Features:")
for i in range(X.shape[1]):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# Plotando a importância das features
plt.figure(figsize=(10, 6))
plt.title("Importância das Features")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
