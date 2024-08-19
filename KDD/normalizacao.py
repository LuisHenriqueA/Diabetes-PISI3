import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Forneça o caminho do arquivo CSV
caminho_arquivo = 'KDD/dfCleaned.csv'

# Carregar o dataframe a partir do arquivo CSV
df_cleaned = pd.read_csv(caminho_arquivo, encoding='utf-8', low_memory=False)

# Inicializar o MinMaxScaler
scaler = MinMaxScaler()

# Normalizar as colunas específicas e armazená-las em um novo dataframe
colunas_para_normalizar = [
    'GenHlth', 'Age', 'MentHlth', 'PhysHlth',
    'Income', 'Education', 'BMI', 'Diabetes_012'
]

df_normalizado = pd.DataFrame()

for coluna in colunas_para_normalizar:
    df_normalizado[coluna + '_normalized'] = scaler.fit_transform(df_cleaned[[coluna]])

# Exibir o dataframe normalizado
print(df_normalizado)
