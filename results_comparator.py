import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_results(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def plot_results(root_path, file_paths):
    # Crear una figura con múltiples subgráficas
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Gráfica 1: val_aps vs Epoch
    axs[0, 0].set_title('APs Validación por Época')
    axs[0, 0].set_xlabel('Época')
    axs[0, 0].set_ylabel('AP')

    # Gráfica 2: new_nodes_val_aps vs Epoch
    axs[0, 1].set_title('APs Validación nuevos nodos por Época')
    axs[0, 1].set_xlabel('Época')
    axs[0, 1].set_ylabel('AP')

    # Gráfica 3: train_losses vs Epoch
    axs[1, 0].set_title('Pérdida de entrenamiento por Época')
    axs[1, 0].set_xlabel('Época')
    axs[1, 0].set_ylabel('Pérdida')

    # Gráfica 4: epoch_times vs Epoch
    axs[1, 1].set_title('Tiempo por Época')
    axs[1, 1].set_xlabel('Época')
    axs[1, 1].set_ylabel('Tiempo')

    for i, file_path in enumerate(file_paths):
        # Cargar los resultados
        results = load_results(file_path)

        # Extraer los datos
        val_aps = results["val_aps"]
        new_nodes_val_aps = results["new_nodes_val_aps"]
        train_losses = results["train_losses"]
        epoch_times = results["epoch_times"]
    
        # Gráfica 1: val_aps vs Epoch
        axs[0, 0].plot(val_aps, label="APs Validación Ejecución - {}".format(i))
        axs[0, 0].legend()
    
        # Gráfica 2: new_nodes_val_aps vs Epoch
        axs[0, 1].plot(new_nodes_val_aps, label="APs Validación nuevos nodos Ejecución - {}".format(i))
        axs[0, 1].legend()
        
        # Gráfica 3: train_losses vs Epoch
        axs[1, 0].plot(train_losses, label="Pérdida de entrenamiento Ejecución - {}".format(i))
        axs[1, 0].legend()
        
        # Gráfica 4: epoch_times vs Epoch
        axs[1, 1].plot(epoch_times, label="Tiempo por Época Ejecución - {}".format(i))
        axs[1, 1].legend()

    # Ajustar el layout
    plt.tight_layout()

    # Guardar gráfica
    plt.savefig(root_path + "Resultados Wikipedia.png", dpi=300)
    
    # Mostrar la gráfica
    plt.show()


def evolucion_nodos(graph_df):
    # Inicializar sets para guardar nodos únicos que ya hemos visto
    seen_u = set()
    seen_i = set()

    # Lista para almacenar los resultados
    results = []

    for batch, group in graph_df.groupby('batch'):
        # Contar nodos únicos en el batch actual
        new_u = group[~group['u'].isin(seen_u)]['u'].nunique()
        new_i = group[~group['i'].isin(seen_i)]['i'].nunique()
        
        # Actualizar los sets de nodos vistos
        seen_u.update(group['u'])
        seen_i.update(group['i'])
        
        # Obtener el rango de timestamps
        ts_min = group['ts'].min()
        ts_max = group['ts'].max()
        
        # Agregar el resultado actual a la lista
        results.append({
            'batch': batch,
            'new_u': new_u,
            'new_i': new_i,
            'ts_min': ts_min,
            'ts_max': ts_max
        })

    # Convertir la lista de resultados en un DataFrame
    result_df = pd.DataFrame(results)

    return result_df

def evolucion_mensajes(graph_df):
    # Número de mensajes por nodo único
    # Agrupar por batch y contar las ocurrencias de cada nodo "u"
    count_u_per_batch = graph_df.groupby(['batch', 'u']).size().reset_index(name='count_u')

    # Agrupar por batch y contar las ocurrencias de cada nodo "i"
    count_i_per_batch = graph_df.groupby(['batch', 'i']).size().reset_index(name='count_i')

    # Obtener el nodo "u" con máximas ocurrencias en cada batch
    max_u_per_batch = count_u_per_batch.loc[count_u_per_batch.groupby('batch')['count_u'].idxmax()].reset_index(drop=True)

    # Obtener el nodo "i" con máximas ocurrencias en cada batch
    max_i_per_batch = count_i_per_batch.loc[count_i_per_batch.groupby('batch')['count_i'].idxmax()].reset_index(drop=True)

    return count_u_per_batch, count_i_per_batch, max_u_per_batch, max_i_per_batch


def grafica_evolucion_nodos(evol_df):
    # Crear la figura y los ejes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Crear un gráfico de barras para los nuevos nodos
    ax1.bar(evol_df['batch'] - 0.2, evol_df['new_u'], width=0.4, label='Nuevos u', align='center')
    ax1.bar(evol_df['batch'] + 0.2, evol_df['new_i'], width=0.4, label='Nuevos i', align='center')

    # Etiquetas y título para el gráfico de barras
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Número de Nuevos Nodos')
    ax1.set_title('Nuevos Nodos por Batch')
    ax1.legend(loc='upper left')

    # Crear un segundo eje y para el rango de timestamps
    ax2 = ax1.twinx()
    ax2.plot(evol_df['batch'], evol_df['ts_min'], color='green', marker='o', linestyle='dashed', linewidth=2, label='ts_min')
    ax2.plot(evol_df['batch'], evol_df['ts_max'], color='red', marker='o', linestyle='dashed', linewidth=2, label='ts_max')

    # Etiquetas y título para el gráfico de líneas
    ax2.set_ylabel('Timestamps')
    ax2.legend(loc='upper right')

    # Mostrar el gráfico
    plt.show()

def grafica_evolucion_mensajes(max_u_per_batch, max_i_per_batch):
    # Crear dos gráficos uno al lado del otro
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 fila, 2 columnas

    # Graficar los datos
    # Graficar max_u_per_batch
    axs[0].bar(max_u_per_batch['batch'], max_u_per_batch['count_u'], color='blue')
    axs[0].set_title('Gráfico 1: Count_u por Batch')
    axs[0].set_xlabel('Batch')
    axs[0].set_ylabel('Count_u')

    # Graficar max_i_per_batch
    axs[1].bar(max_i_per_batch['batch'], max_i_per_batch['count_i'], color='red')
    axs[1].set_title('Gráfico 2: Count_i por Batch')
    axs[1].set_xlabel('Batch')
    axs[1].set_ylabel('Count_i')

    plt.tight_layout()  # Ajusta automáticamente el diseño de las figuras
    plt.show()

file_paths = []
root_path = "./results/link_prediction/reddit-tgn/"

for i in range(10):
    if i == 0:
        file_paths.append(root_path + "tgn-attn-reddit-tgn.pkl")
    else:
        file_paths.append(root_path + "tgn-attn-reddit-tgn_" + str(i) + ".pkl")

plot_results(root_path, file_paths)

graph_df = pd.read_csv('./data/wikipedia-tgn/ml_wikipedia_df.csv')

graph_df['batch'] = np.arange(len(graph_df)) // 200

evol_nodos = evolucion_nodos(graph_df)

count_u_per_batch, count_i_per_batch, max_u_per_batch, max_i_per_batch = evolucion_mensajes(graph_df)

mean_u = max_u_per_batch['count_u'].mean()
mean_i = max_i_per_batch['count_i'].mean()