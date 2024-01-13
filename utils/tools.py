import yaml
from typing import Dict
import torch
from utils import constants
import numpy as np
import random
def load_model_configs(config_path:str) -> Dict:
    '''
    Load model parameters from config.yaml.
    '''
    with open(config_path,'r') as f:

        config = yaml.safe_load(f)

    return config

def calculate_adjacency(dataset_name:str) -> torch.Tensor:

    '''
    Calculate normalized adjacency.

    dataset_name: `ntu` or `...`
    '''

    paris = constants.paris[dataset_name]
    total = 25 if dataset_name == 'ntu' else 20 # TODO: Finish other dataset.
    adjacency = torch.zeros((total, total))

    # Calculate adjacency matrix.
    for i, j in paris:

        adjacency[i-1,j-1] = 1

    # A_hat = A + I.
    adjacency += torch.eye(total)  

    # Normalization.
    D = torch.pow(adjacency.sum(dim = 1), -0.5).flatten()
    D_diag = torch.diag(D)
    adjacency_normalized = D_diag.T@adjacency@D_diag

    return adjacency_normalized


def init_seed(seed:int) -> None:
    '''
    Set all seed.
    '''
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True