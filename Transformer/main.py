import os
import sys

# ensure the directory containing this file is on sys.path so local packages (training, models) can be imported
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from training.train import train_model
from training.hyperparameters import feature_dim, seq_len, number_heads, num_layers, dim_feedforward, dropout
from models.Transformer_v1 import Transformer_V1

model = Transformer_V1(feature_dim, seq_len, number_heads, num_layers, dim_feedforward, dropout)

train_model(model, project_root)