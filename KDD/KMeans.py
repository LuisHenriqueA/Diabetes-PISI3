import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Carregar os dados
file_path = 'DiabetesDataSet/diabetes_012_health_indicators_BRFSS2015.csv'
df = pd.read_csv(file_path)

# Selecionar as colunas relevantes para estilo de vida e condições diabéticas
features = df[['Smoker', 'HvyAlcoholConsump', 'Fruits', 'Veggies', 'GenHlth', 'MentHlth']]

# Verificar a quantidade de dados
print(f"Total de pontos de dados: {len(df)}")
print(f"Colunas disponíveis para visualização: {features.columns.tolist()}")

# Normalizar os dados
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Encontrar o número ideal de clusters usando o método do cotovelo
inertia = []
k_range = range(1, 11)  # Testar para 1 a 10 clusters
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Com base no gráfico, defina o número de clusters
optimal_clusters = 5  # Ajuste com base no gráfico do cotovelo

# Aplicar KMeans com o número ideal de clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Verificar a distribuição dos clusters
print("Distribuição dos Clusters:")
print(df['cluster'].value_counts())

# Analisar os clusters
cluster_means = df.groupby('cluster').mean()
print("Médias dos Atributos por Cluster:")
print(cluster_means)

# Reduzir a dimensionalidade para visualização dos clusters
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)
df_pca = pd.DataFrame(features_pca, columns=['PC1', 'PC2'])
df_pca['cluster'] = df['cluster']

# Plotar os clusters com base nas componentes principais
plt.figure(figsize=(12, 8))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['cluster'], cmap='viridis', alpha=0.5, edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters com Base nas Componentes Principais')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Opcional: Salvar o DataFrame com os clusters atribuídos
df.to_csv('dados_com_clusters_estilo_de_vida.csv', index=False)