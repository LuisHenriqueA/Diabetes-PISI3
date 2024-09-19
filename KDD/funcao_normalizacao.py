from sklearn.preprocessing import MinMaxScaler

def normalizar_coluna(df, nome_coluna):
    # Inicializar o MinMaxScaler
    scaler = MinMaxScaler()
    
    # Selecionar a coluna especificada
    coluna = df[[nome_coluna]]
    
    # Normalizar a coluna
    coluna_normalizada = scaler.fit_transform(coluna)
    
    # Retornar a coluna normalizada
    return coluna_normalizada

# Exemplo de uso
# df_cleaned['GenHlth_normalizada'] = normalizar_coluna(df_cleaned, 'GenHlth')
