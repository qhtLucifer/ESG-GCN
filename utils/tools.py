import yaml
from typing import Dict
import torch
from utils import constants
import numpy as np
import random

from torch import optim

from feeders.feeder_ntu import  Feeder
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


def load_data(args)->Dict:
    '''
    Return dict[train_dataloader, test_dataloader]
    '''

    data_path = f'data/{args.dataset_name}/{args.split_type}_aligned.npz'
    data_loader = dict()
    data_loader['train'] = torch.utils.data.DataLoader(

            dataset=Feeder(data_path=data_path,
                split='train',
                p_interval=[0.5, 1],
                random_rot=args.random_rot,
                sort=False,
                window_size=args.window_size,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=init_seed)

    data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(
                data_path=data_path,
                split='test',
                p_interval=[0.95],
                window_size=args.window_size,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=init_seed)
    
    return data_loader

def load_optimizer(model, args) -> optim.Optimizer:

    if args.optimizer == 'SGD':
            
            optimizer = optim.SGD(
               model.parameters(),
                lr=args.base_learning_rate,
                momentum=0.9,
                nesterov=args.nesterov,
                weight_decay=args.weight_decay)
            
    elif args.optimizer == 'Adam':
            
            optimizer = optim.Adam(
              model.parameters(),
                lr=args.base_learning_rate,
                weight_decay=args.weight_decay)
            
    else:
            
            raise ValueError()
    

    return optimizer

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

