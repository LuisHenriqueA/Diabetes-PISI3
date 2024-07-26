import pandas as pd

# Caminho para o arquivo CSV
csv_file_path = 'DiabetesDataSet/diabetes_012_health_indicators_BRFSS2015.csv'

# Caminho para o arquivo Parquet de saída
parquet_file_path = 'DiabetesDataSet/diabetes_012_health_indicators_BRFSS2015.parquet'

df = pd.read_csv(csv_file_path)

df.to_parquet(parquet_file_path, engine='pyarrow')