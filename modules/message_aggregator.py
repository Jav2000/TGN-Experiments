from collections import defaultdict
import torch
import numpy as np
from torch import nn


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

class RNNMessageAggregator(MessageAggregator):
  def __init__(self, device, raw_message_dimension):
    super(RNNMessageAggregator, self).__init__(device)

    self.raw_message_dimension = raw_message_dimension
    self.message_aggregator = nn.RNNCell(input_size=raw_message_dimension, hidden_size=raw_message_dimension)
  
  def aggregate(self, node_ids, messages):
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        to_update_node_ids.append(node_id)
        
        hidden_state = nn.Parameter(torch.zeros(self.raw_message_dimension).to(self.device),
                               requires_grad=False)

        for message in messages[node_id]:
          hidden_state = self.message_aggregator(message[0].squeeze(0), hidden_state)

        unique_messages.append(hidden_state)
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps
  
class GRUMessageAggregator(MessageAggregator):
  def __init__(self, device, raw_message_dimension):
    super(GRUMessageAggregator, self).__init__(device)

    self.raw_message_dimension = raw_message_dimension
    self.message_aggregator = nn.GRUCell(input_size=raw_message_dimension, hidden_size=raw_message_dimension)
  
  def aggregate(self, node_ids, messages):
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        to_update_node_ids.append(node_id)
        
        hidden_state = nn.Parameter(torch.zeros(self.raw_message_dimension).to(self.device),
                               requires_grad=False)

        for message in messages[node_id]:
          hidden_state = self.message_aggregator(message[0].squeeze(0), hidden_state)

        unique_messages.append(hidden_state.detach())
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps
  
class GLUMessageAggregator(MessageAggregator):
  def __init__(self, device, raw_message_dimension):
    super(GRUMessageAggregator, self).__init__(device)

    self.raw_message_dimension = raw_message_dimension
    self.message_aggregator = nn.GRUCell(input_size=raw_message_dimension, hidden_size=raw_message_dimension)
  
  def aggregate(self, node_ids, messages):
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        to_update_node_ids.append(node_id)
        
        hidden_state = nn.Parameter(torch.zeros(self.raw_message_dimension).to(self.device),
                               requires_grad=False)

        for message in messages[node_id]:
          hidden_state = self.message_aggregator(message[0].squeeze(0), hidden_state)

        unique_messages.append(hidden_state.detach())
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

def get_message_aggregator(aggregator_type, device, raw_message_dimension=None):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  elif aggregator_type == "rnn":
    return RNNMessageAggregator(device=device, raw_message_dimension=raw_message_dimension)
  elif aggregator_type == "gru":
    return GRUMessageAggregator(device=device, raw_message_dimension=raw_message_dimension)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
