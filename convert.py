import pandas as pd

# Caminho para o arquivo CSV
csv_file_path = 'KDD/dfCleaned.csv'

# Caminho para o arquivo Parquet de saída
parquet_file_path = 'KDD/dfCleaned.parquet'

df = pd.read_csv(csv_file_path)

df.to_parquet(parquet_file_path, engine='pyarrow')