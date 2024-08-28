import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
base = pd.read_csv('KDD/dfCleaned.csv') #alterar
base.head()
#Separando em X e Y 
X = base.drop('DiffWalk',axis=1)
y = base.DiffWalk

# Substituir valores numéricos por labels
labels = {0.0: "Sem dificuldade", 1.0: "Com dificuldade"}
y_labels = y.replace(labels)

# Contar a frequência dos valores
value_counts = y_labels.value_counts()

# Plotar gráfico de barras
value_counts.plot(kind='bar', color=['blue', 'orange'])

# Adicionar título e labels
plt.title('Distribuição de Dificuldades')
plt.xlabel('Categoria')
plt.ylabel('Contagem')

# Mostrar gráfico
plt.show()
# Importando o train_test_split
from sklearn.model_selection import train_test_split
# Fazendo a separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                    random_state=42,stratify=y)
# Verificando a proporção na base de treino
(y_train.value_counts()/y_train.shape[0])*100
# Treino
knn = KNeighborsClassifier(n_neighbors=3)

# Treinando o modelo
knn.fit(X_train, y_train)

# Fazendo previsões nos dados de teste
y_pred = knn.predict(X_test)

# Avaliando o desempenho do modelo nos dados de teste
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Acurácia KNN desbalanceado: {accuracy:.2f}")
print("Relatório de Classificação:")
print(report)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Testando um modelo Naive Bayes
nb = GaussianNB()
nb.fit(X_test, y_test)
y_pred_nb = nb.predict(X_test)

# Avaliando o desempenho
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)

print(f"Acurácia (Naive Bayes) desbalanceado: {accuracy_nb:.2f}")
print(report_nb)
from sklearn.ensemble import RandomForestClassifier

# Testando um modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Avaliando o desempenho
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print(f"Acurácia (Random Forest) desbalanceado: {accuracy_rf:.2f}")
print(report_rf)
from imblearn.under_sampling import RandomUnderSampler
# Definindo o RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
# Aplicando para X e y
X_res, y_res = rus.fit_resample(X_train, y_train)
y_res.value_counts()
# Criando o modelo KNN
knn_balanced = KNeighborsClassifier(n_neighbors=3)

# Treinando o modelo
knn_balanced.fit(X_res, y_res)

# Fazendo previsões nos dados de teste
y_pred2 = knn_balanced.predict(X_test)

# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred2)
report = classification_report(y_test, y_pred2)

print(f"Acurácia knn balanceado: {accuracy:.2f}")
print("Relatório de Classificação balanceado:")
print(report)
from sklearn.ensemble import RandomForestClassifier

# Testando um modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_res, y_res)
y_pred_rf = rf.predict(X_test)

# Avaliando o desempenho
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print(f"Acurácia (Random Forest) balanceado: {accuracy_rf:.2f}")
print("Relatório de Classificação balanceado:")
print(report_rf)
# Testando um modelo Naive Bayes
nb_balanced = GaussianNB()
nb_balanced.fit(X_res, y_res)
y_pred_nb_balanced = nb_balanced.predict(X_test)

# Avaliando o desempenho
accuracy_nb_balanced = accuracy_score(y_test, y_pred_nb_balanced)
report_nby_pred_nb_balanced = classification_report(y_test, y_pred_nb_balanced)

print(f"Acurácia (Naive Bayes) balanceado: {accuracy_nb_balanced:.2f}")
print("Relatório de Classificação balanceado:")
print(report_nby_pred_nb_balanced)

