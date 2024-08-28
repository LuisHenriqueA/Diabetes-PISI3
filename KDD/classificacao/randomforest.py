import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Carregando a base de dados
base = pd.read_csv('/Users/alanalins/pisi3/Diabetes-PISI3/KDD/dfCleaned.csv')  # alterar o caminho se necessário
base.head()

# Definindo variáveis independentes e dependentes
X = base.drop('DiffWalk', axis=1)
y = base.DiffWalk

# Dividindo os dados em treino e teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Aplicando SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Padronizando os dados de treino e teste
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)
X_test_scaled = scaler.transform(X_test)

# Definindo o modelo Random Forest
rf = RandomForestClassifier(random_state=42)

# Definindo os hiperparâmetros para o GridSearch
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Usando validação cruzada com GridSearch para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_res, y_res)

# Melhor modelo encontrado
best_rf = grid_search.best_estimator_
print(f"Melhores Hiperparâmetros: {grid_search.best_params_}")

# Previsões nos dados de treino usando o melhor modelo
y_pred_train_cv = cross_val_predict(best_rf, X_res, y_res, cv=5)

# Treinando o melhor modelo nos dados de treino balanceados
best_rf.fit(X_res, y_res)

# Previsões nos dados de teste usando o melhor modelo
y_pred_test = best_rf.predict(X_test_scaled)

# Avaliando o desempenho do modelo nos dados de treino (com previsões de validação cruzada)
accuracy_train_cv = accuracy_score(y_res, y_pred_train_cv)
report_train_cv = classification_report(y_res, y_pred_train_cv)

print(f"Acurácia no conjunto de treino (validação cruzada): {accuracy_train_cv:.2f}")
print("Relatório de Classificação - Conjunto de Treino (validação cruzada):")
print(report_train_cv)

# Avaliando o desempenho do modelo nos dados de teste
accuracy_test = accuracy_score(y_test, y_pred_test)
report_test = classification_report(y_test, y_pred_test)

print(f"Acurácia no conjunto de teste: {accuracy_test:.2f}")
print("Relatório de Classificação - Conjunto de Teste:")
print(report_test)
