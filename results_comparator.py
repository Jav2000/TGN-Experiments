import pickle
import pandas as pd



with open('./results/tgn-attn_1.pkl', 'rb') as archivo:
    results_1 = pickle.load(archivo)

with open('./results/tgn-attn_1.pkl', 'rb') as archivo:
    results_2 = pickle.load(archivo)

df_results_1 = pd.DataFrame(results_1)
df_results_2 = pd.DataFrame(results_2)
# Imprimir el DataFrame
print(df)