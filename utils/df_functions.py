'''Funções recorrentes, leitura de arquivos CSV e Parquet'''

import pandas as pd


def __read_csv(arquivo,encoding,low_memory=False):
    try:
        df = pd.read_csv(arquivo, sep=',', encoding=encoding, low_memory=low_memory)
    except pd.errors.ParserError:
        df = pd.read_csv(arquivo, sep=';', encoding=encoding, low_memory=low_memory)
    return df

def read_csv (df_name, extension='csv', encoding='utf-8', low_memory=False):
    arquivo = f'DiabetesDataSet/{df_name}.{extension}'
    return __read_csv(arquivo, encoding=encoding, low_memory=low_memory)

def read_parquet(df_name, extension="parquet"):
    arquivo = f'{df_name}.{extension}'
    return pd.read_parquet(arquivo)