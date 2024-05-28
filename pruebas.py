import pandas as pd
import numpy as np
import random
import dgl
import torch
import torch.nn

def simplificar_wikipedia(graph_df, edge_feat):
    '''
    Simplificamos el conjunto de datos de Wikipedia para ahorrar en el tiempo de las pruebas.

    Se eliminan un 50% de los usuarios y las páginas correspondientes, dejando un total de:

        * 4113/8227 usuario y 952/1000 páginas
    '''
    unique_u = graph_df['u'].unique()

    num_nodes_u_to_remove = int(len(unique_u) * 0.5)

    # Establecer una semilla para reproducibilidad
    random.seed(1) 

    random_selection = random.sample(list(unique_u), num_nodes_u_to_remove)

    df_filtered = graph_df[graph_df['u'].isin(random_selection)]

    # Reemplazar los valores de 'u' e 'i' con nuevos índices comenzando desde 1
    new_index = {old_index: new_index + 1 for new_index, old_index in enumerate(sorted(set(df_filtered['u']).union(df_filtered['i'])))}
    df_reindexed = df_filtered.replace({'u': new_index, 'i': new_index}).sort_values(by='ts')

    new_edge_feat = edge_feat[df_reindexed['idx']]

    # Reemplazar los valores del index e 'idx' con nuevos índices comenzando desde 0
    df_reindexed.reset_index(drop=True, inplace=True)
    df_reindexed.iloc[:, 0]= df_reindexed.index
    df_reindexed['idx'] = df_reindexed.index

    return df_reindexed, new_edge_feat


# graph_df = pd.read_csv('./data/wikipedia-tgn/ml_wikipedia_df.csv')
# edge_feat = np.load('./data/wikipedia-tgn/ml_wikipedia_edge_feat.npy')

# print("Número de usuarios en el grafo original: ")
# print(len(graph_df['u'].unique()))
# print("Número de items en el grafo original: ")
# print(len(graph_df['i'].unique()))

# df_reindexed, new_edge_feat = simplificar_wikipedia(graph_df, edge_feat)

# print("Número de usuarios en el grafo simplificado: ")
# print(len(df_reindexed['u'].unique()))
# print("Número de items en el grafo simplificado: ")
# print(len(df_reindexed['i'].unique()))

# df_reindexed.to_csv('./data/wikipedia-simplificada/ml_wikipedia_simplificada_df.csv', index=False)
# np.save('./data/wikipedia-simplificada/ml_wikipedia_simplificada_edge_feat.npy', new_edge_feat)

# import pickle

# # Ruta al archivo .pkl
# file_path = './results/link_prediction/wikipedia-simplificada/Original WorkFlow/tgn-attn/tgn-attn.pkl'

# # Abrir el archivo .pkl
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Mostrar el contenido del archivo
# print(data)