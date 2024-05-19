# Experimentos con Temporal Graph Networks

## Introducción

Partiendo del artículo [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637) y del repositorio https://github.com/twitterresearch/tgn, la propuesta consiste en una evaluación experimental de las Temporal Graph
Networks definidas en dicho artículo, analizando las líneas futuras propuestas y valorando
los resultados obtenidos.

## Ejecución de los experimentos

### Requerimientos

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

### Conjuntos de datos y pre-procesado

#### Conjuntos de datos de TGN

##### Descargar los datos
Se pueden descargar los conjuntos de datos de wikipedia y reddit desde [aquí](http://snap.stanford.edu/jodie/) y se deben almacenar en las carpetas
```data/tgn_wikipedia``` y ```data/tgn_reddit``` respectivamente.

#### Pre-procesamiento de los datos
Se emplean archivos .npy para guardar los datos creados. Si las características de los nodos o aristas están vacías, se rellenarán con 0's.
```{bash}
python3 utils/tgn_preprocess_data.py --data wikipedia --bipartite
python3 utils/tgn_preprocess_data.py --data reddit --bipartite
```

### Entrenamiento del modelo

Para la predicción de enlaces:
```{bash}
# TGN-attn: Supervised learning on the wikipedia dataset
python3 tgn_link_prediction.py --use_memory --prefix tgn-attn --n_runs 10

# TGN-attn-reddit: Supervised learning on the reddit dataset
python tgn_link_prediction.py -d reddit --use_memory --prefix tgn-attn-reddit --n_runs 10
```

Para la clasificación de nodos(se requiere el modelo entrenado en la tarea de predicción de enlaces):
```{bash}
# TGN-attn: self-supervised learning on the wikipedia dataset
python3 tgn_node_classification.py --use_memory --prefix tgn-attn --n_runs 10

# TGN-attn-reddit: self-supervised learning on the reddit dataset
python3 tgn_node_classification.py -d reddit --use_memory --prefix tgn-attn-reddit --n_runs 10
```

### JODIE y DyRep

```{bash}
### Predicción de enlaces en Wikipedia

# Jodie
python3 tgn_link_prediction.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --n_runs 10

# DyRep
python3 tgn_link_prediction.py --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --n_runs 10


### Predicción de enlaces en Reddit

# Jodie
python3 tgn_link_prediction.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn_reddit --n_runs 10

# DyRep
python3 tgn_link_prediction.py -d reddit --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn_reddit --n_runs 10


### Clasificación de nodos en Wikipedia

# Jodie
python3 tgn_node_classification.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --n_runs 10

# DyRep
python3 tgn_node_classification.py --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --n_runs 10


### Clasificación de nodos en Reddit

# Jodie
python3 tgn_node_classification.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn_reddit --n_runs 10

# DyRep
python3 tgn_node_classification.py -d reddit --use_memory --memory_updater rnn  --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn_reddit --n_runs 10
```


### Ablation Study
Commands to replicate all results in the ablation study over different modules:
```{bash}
# TGN-2l
python train_self_supervised.py --use_memory --n_layer 2 --prefix tgn-2l --n_runs 10 

# TGN-no-mem
python train_self_supervised.py --prefix tgn-no-mem --n_runs 10 

# TGN-time
python train_self_supervised.py --use_memory --embedding_module time --prefix tgn-time --n_runs 10 

# TGN-id
python train_self_supervised.py --use_memory --embedding_module identity --prefix tgn-id --n_runs 10

# TGN-sum
python train_self_supervised.py --use_memory --embedding_module graph_sum --prefix tgn-sum --n_runs 10

# TGN-mean
python train_self_supervised.py --use_memory --aggregator mean --prefix tgn-mean --n_runs 10
```


#### General flags

```{txt}
optional arguments:
  -d DATA, --data DATA         Data sources to use (wikipedia or reddit)
  --bs BS                      Batch size
  --prefix PREFIX              Prefix to name checkpoints and results
  --n_degree N_DEGREE          Number of neighbors to sample at each layer
  --n_head N_HEAD              Number of heads used in the attention layer
  --n_epoch N_EPOCH            Number of epochs
  --n_layer N_LAYER            Number of graph attention layers
  --lr LR                      Learning rate
  --patience                   Patience of the early stopping strategy
  --n_runs                     Number of runs (compute mean and std of results)
  --drop_out DROP_OUT          Dropout probability
  --gpu GPU                    Idx for the gpu to use
  --node_dim NODE_DIM          Dimensions of the node embedding
  --time_dim TIME_DIM          Dimensions of the time embedding
  --use_memory                 Whether to use a memory for the nodes
  --embedding_module           Type of the embedding module
  --message_function           Type of the message function
  --memory_updater             Type of the memory updater
  --aggregator                 Type of the message aggregator
  --memory_update_at_the_end   Whether to update the memory at the end or at the start of the batch
  --message_dim                Dimension of the messages
  --memory_dim                 Dimension of the memory
  --backprop_every             Number of batches to process before performing backpropagation
  --different_new_nodes        Whether to use different unseen nodes for validation and testing
  --uniform                    Whether to sample the temporal neighbors uniformly (or instead take the most recent ones)
  --randomize_features         Whether to randomize node features
  --dyrep                      Whether to run the model as DyRep
```

## TODOs 
* Make code memory efficient: for the sake of simplicity, the memory module of the TGN model is 
implemented as a parameter (so that it is stored and loaded together of the model). However, this 
does not need to be the case, and 
more efficient implementations which treat the models as just tensors (in the same way as the 
input features) would be more amenable to large graphs.

## Cite us

```bibtex
@inproceedings{tgn_icml_grl2020,
    title={Temporal Graph Networks for Deep Learning on Dynamic Graphs},
    author={Emanuele Rossi and Ben Chamberlain and Fabrizio Frasca and Davide Eynard and Federico 
    Monti and Michael Bronstein},
    booktitle={ICML 2020 Workshop on Graph Representation Learning},
    year={2020}
}
```


