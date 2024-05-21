import numpy as np
import pandas as pd
import pickle

# Cargar el archivo .npy
data = np.load('data/tgbl-coin/ml_coin_node_feat.npy')
with open('data/tgbl-coin/ml_coin_node_dict.pkl', 'rb') as archivo:
    mi_diccionario_cargado = pickle.load(archivo)

print(type(dict))

# Convertir el array de numpy en un DataFrame de pandas
df = pd.DataFrame(data)

contador = 0
for clave, valor in mi_diccionario_cargado.items():
    print(f'Clave: {clave} -> Valor: {valor}')
    contador += 1
    if contador >= 10:
        break

# Mostrar el DataFrame
print(df[0:10])
