import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle
import csv
import tqdm

def preprocess_wikipedia(data_name):
    '''
    [u, i, ts, label, feats]
    '''
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                        'i': i_list,
                        'ts': ts_list,
                        'label': label_list,
                        'idx': idx_list}), np.array(feat_l), None

def preprocess_review_coin(data_name):
  '''
  [ts, u, i, weight]
  '''
  u_list, i_list, ts_list, idx_list, feat_list = [], [], [], [], []
  
  node_dict = {}
  nodes_id = 0

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')

      ts = float(e[0])
      if "coin" in data_name:
        u = e[1]
        i = e[2]
      else:
        u = int(e[1])
        i = int(e[2])
      
      if u not in node_dict:
          node_dict[u] = nodes_id
          nodes_id += 1
      if i not in node_dict:
          node_dict[i] = nodes_id
          nodes_id += 1

      u_list.append(node_dict[u])
      i_list.append(node_dict[i])
      ts_list.append(ts)
      idx_list.append(idx)

      feat_list.append(np.zeros(1))


  return pd.DataFrame({"u": u_list,
                       "i": i_list,
                       "ts": ts_list,
                       "idx": idx_list}), np.array(feat_list), node_dict


def preprocess_comment(data_name):
  '''
  [ts, u, i, subrredit, num_words, score]
  '''
  u_list, i_list, ts_list, idx_list, feat_list = [], [], [], [], []
  
  # Del código TGB
  max_words = 500

  node_dict = {}
  nodes_id = 0
  

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')

      ts = float(e[0])
      u = e[1]
      if u not in node_dict:
          node_dict[u] = nodes_id
          nodes_id += 1

      i = e[2]
      if i not in node_dict:
          node_dict[i] = nodes_id
          nodes_id += 1

      u_list.append(node_dict[u])
      i_list.append(node_dict[i])
      ts_list.append(ts)
      idx_list.append(idx)

      feat_list.append(np.array([(float(e[4])/max_words)]))


  return pd.DataFrame({"u": u_list,
                       "i": i_list,
                       "ts": ts_list,
                       "idx": idx_list}), np.array(feat_list), node_dict
  
def preprocess_flight(data_name):
  '''
  [ts, u, i, callsing, typecode]
  '''
  u_list, i_list, ts_list, idx_list, feat_list = [], [], [], [], []

  node_dict = {}
  nodes_id = 0
  

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')

      ts = float(e[0])
      u = e[1]
      if u not in node_dict:
          node_dict[u] = nodes_id
          nodes_id += 1

      i = e[2]
      if i not in node_dict:
          node_dict[i] = nodes_id
          nodes_id += 1

      u_list.append(node_dict[u])
      i_list.append(node_dict[i])
      ts_list.append(ts)
      idx_list.append(idx)

      # Fix size to 8 with !
      if len(e[3]) == 0:
        e[3] = "!!!!!!!!"
      while len(e[3]) < 8:
        e[3] += "!"

      if len(e[4]) == 0:
        e[4] = "!!!!!!!!"
      while len(e[4]) < 8:
        e[4] += "!"
      if len(e[4]) > 8:
        e[4] = "!!!!!!!!"

      feat_str = e[3] + e[4]

      feat_list.append(convert_str2int(feat_str))


  return pd.DataFrame({"u": u_list,
                       "i": i_list,
                       "ts": ts_list,
                       "idx": idx_list}), np.array(feat_list), node_dict

def reindex(df, node_dict, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    new_dict = None
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    new_dict = {key: value + 1 for key, value in node_dict.items()}

  return new_df, new_dict

def convert_str2int(in_str: str) -> np.ndarray:
    """
    convert strings to vectors of integers based on individual character
    each letter is converted as follows, a=10, b=11
    numbers are still int
    Parameters:
        in_str: an input string to parse
    Returns:
        out: a numpy integer array
    """
    out = []
    for element in in_str:
        if element.isnumeric():
            out.append(element)
        elif element == "!":
            out.append(-1)
        else:
            out.append(ord(element.upper()) - 44 + 9)
    out = np.array(out, dtype=np.float32)
    return out

def process_flight_node_feat(fname: str, node_ids):
    """
    1. need to have the same node id as csv_to_pd_data
    2. process the various node features into a vector
    3. return a numpy array of node features with index corresponding to node id

    airport_code,type,continent,iso_region,longitude,latitude
    type: onehot encoding
    continent: onehot encoding
    iso_region: alphabet encoding same as edge feat
    longitude: float divide by 180
    latitude: float divide by 90
    """
    feat_size = 20
    node_feat = np.zeros((len(node_ids), feat_size))
    type_dict = {}
    type_idx = 0
    continent_dict = {}
    cont_idx = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # airport_code,type,continent,iso_region,longitude,latitude
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                code = row[0]
                if code not in node_ids:
                    continue
                else:
                    node_id = node_ids[code]
                    airport_type = row[1]
                    if airport_type not in type_dict:
                        type_dict[airport_type] = type_idx
                        type_idx += 1
                    continent = row[2]
                    if continent not in continent_dict:
                        continent_dict[continent] = cont_idx
                        cont_idx += 1

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # airport_code,type,continent,iso_region,longitude,latitude
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                code = row[0]
                if code not in node_ids:
                    continue
                else:
                    node_id = node_ids[code]
                    airport_type = type_dict[row[1]]
                    type_vec = np.zeros(type_idx)
                    type_vec[airport_type] = 1
                    continent = continent_dict[row[2]]
                    cont_vec = np.zeros(cont_idx)
                    cont_vec[continent] = 1
                    while len(row[3]) < 7:
                        row[3] += "!"
                    iso_region = convert_str2int(row[3])  # numpy float array
                    lng = float(row[4])
                    lat = float(row[5])
                    coor_vec = np.array([lng, lat])
                    final = np.concatenate(
                        (type_vec, cont_vec, iso_region, coor_vec), axis=0
                    )
                    node_feat[node_id] = final
    return node_feat

def run(data_name, bipartite=True):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = './data/{}/{}_edgelist.csv'.format(data_name, data_name.split('-')[0])
    OUT_DF = './data/{}/ml_{}_df.csv'.format(data_name, data_name.split('-')[0])
    OUT_EDGE_FEAT = './data/{}/ml_{}_edge_feat.npy'.format(data_name, data_name.split('-')[0])
    OUT_NODE_FEAT = './data/{}/ml_{}_node_feat.npy'.format(data_name, data_name.split('-')[0])
    OUT_NODE_DICT = './data/{}/ml_{}_node_dict.pkl'.format(data_name, data_name.split('-')[0])

    if data_name == "wikipedia-tgb":
        df, feat, node_dict = preprocess_wikipedia(PATH)
    elif data_name in ["review-tgb", "coin-tgb"]:
        df, feat, node_dict = preprocess_review_coin(PATH)
    elif data_name == "comment-tgb":
      df, feat, node_dict = preprocess_comment(PATH)
    elif data_name == "flight-tgb":
      df, feat, node_dict = preprocess_flight(PATH)
    else:
      print("Conjunto de datos no existente")
      return

    new_df, new_dict = reindex(df, node_dict, bipartite)
    
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    new_df.to_csv(OUT_DF)
    np.save(OUT_EDGE_FEAT, feat)

    # Se deciden guardar únicamente las características de los nodos del conjunto "flight" ya que no son 0.
    if data_name == "flight":
      # Se calculan las características de los nodos de "flight"
      node_feat = process_flight_node_feat("./data/flight-tgb/airport_node_feat.csv", node_dict)
      np.save(OUT_NODE_FEAT, node_feat)

    if node_dict != None:
      with open(OUT_NODE_DICT, 'wb') as archivo:
        pickle.dump(new_dict, archivo)


parser = argparse.ArgumentParser('Interface for TGB data preprocessing')
parser.add_argument('--data_name', type=str, help='Dataset name (eg. wikipedia, review, coin, comment, flight)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data_name, bipartite=args.bipartite)