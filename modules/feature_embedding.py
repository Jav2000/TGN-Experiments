from torch import nn

class FeatureEmbedding(nn.Module):
  """
  Embedding module for edge features.
  """
  def compute_features(self, raw_features):
    return None
  
class MLPFeatureEmbedding(FeatureEmbedding):
  def __init__(self, raw_features_dimension, features_dimension):
    super(MLPFeatureEmbedding, self).__init__()

    self.mlp = self.layers = nn.Sequential(
      nn.Linear(raw_features_dimension, raw_features_dimension // 2),
      nn.ReLU(),
      nn.BatchNorm1d(raw_features_dimension // 2),
      nn.Dropout(0.2),

      nn.Linear(raw_features_dimension // 2, raw_features_dimension // 4),
      nn.ReLU(),
      nn.BatchNorm1d(raw_features_dimension // 4),
      nn.Dropout(0.2),

      nn.Linear(raw_features_dimension // 4, features_dimension),
    )

  def compute_features(self, raw_features):
    messages = self.mlp(raw_features)

    return messages
  
class IdentityFeatureEmbedding(FeatureEmbedding):
  def compute_features(self, raw_features):
    return raw_features

def get_feature_embedding(module_type, raw_features_dimension, features_dimension=50):
  if module_type == "mlp":
    return MLPFeatureEmbedding(raw_features_dimension, features_dimension)
  elif module_type == "identity":
    return IdentityFeatureEmbedding()