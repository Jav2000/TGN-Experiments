from torch import nn

class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """
  def compute_message(self, raw_messages):
    return None

class MLPMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension):
    super(MLPMessageFunction, self).__init__()

    self.mlp = self.layers = nn.Sequential(
      nn.Linear(raw_message_dimension, raw_message_dimension // 2),
      nn.ReLU(),
      nn.BatchNorm1d(raw_message_dimension // 2),
      nn.Dropout(0.2),

      nn.Linear(raw_message_dimension // 2, raw_message_dimension // 4),
      nn.ReLU(),
      nn.BatchNorm1d(raw_message_dimension // 4),
      nn.Dropout(0.2),

      nn.Linear(raw_message_dimension // 4, message_dimension),
    )

  def compute_message(self, raw_messages):
    messages = self.mlp(raw_messages)

    return messages

class IdentityMessageFunction(MessageFunction):
  def compute_message(self, raw_messages):

    return raw_messages

def get_message_function(module_type, raw_message_dimension, message_dimension):
  if module_type == "mlp":
    return MLPMessageFunction(raw_message_dimension, message_dimension)
  elif module_type == "identity":
    return IdentityMessageFunction()
