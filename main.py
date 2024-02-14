from model.ESRGCN import ESRGCN

from utils.tools import load_data,load_optimizer,AverageMeter
from args import parse_args

from torch.nn import CrossEntropyLoss
import torch

import os 
import wandb 
import logging
from tqdm import tqdm
import math
import numpy as np

import yaml

# TODO: Finish main.py

class Processor():
    '''
    Processor for Skeleton-based action recognition.
    '''

    def __init__(self,args) -> None:
        '''
        Initialization for wandb, model, training and testing.
        '''
        wandb.init(project=args.project_name)

        self.device:str = args.device

        self.model = ESRGCN(args).to(device=self.device)
        self.data = load_data(args)
        self.loss_func = CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.optimizer:torch.optim.Optimizer = load_optimizer(self.model, args)

        self.auto_resume:bool = args.auto_resume
        self.log_dir:str = args.log_dir

        self.best_acc:float = 0
        self.log_acc = [AverageMeter() for _ in range(10)]
        self.loss = AverageMeter()

        self.step:int = 0
        self.max_epoch:int = args.max_epoch
        self.validate_per_epoch:int = args.validate_per_epoch
        self.n_classes:int = args.n_classes

    def train(self, epoch:int) -> None:
        '''
        Train for one epoch.
        '''
        self.model.train()
        self.adjust_learning_rate(epoch)
        data_iter = tqdm(self.data['train'],dynamic_ncols = True)
        for X, y, mask, index in data_iter:
            B:int = X.shape[0]
            T:int = X.shape[2]
            y_hat = self.model(X)
            y = y.to(self.device)

            loss = self.loss_func(y_hat, y)
            _, predicted_labels = torch.max(y_hat, 1)
            self.loss.update(loss.data.item(),B)
            for i, ratio in enumerate([(i+1)/10 for i in range(10)]):

                self.log_acc[i].update((predicted_labels == y.data)\
                                        .view(self.n_classes*B,-1)[:,int(math.ceil(T*ratio))-1].float().mean(), B)
            
            AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
            data_iter.set_description(
                f"[Epoch #{epoch}] "\
                f"AUC:{AUC:.3f}, " \
                f"LOSS:{self.loss.avg:.3f}, " 
            )
            train_logs:dict = {
            "train/LOSS":self.loss.avg,
            "train/AUC":AUC,
            }
            train_logs.update({f"train/ACC_{(i+1)/10}":self.log_acc[i].avg for i in range(10)})
            wandb.log(train_logs)


    def validate(self, epoch:int) -> None:
        """
        Validate for one epoch.
        """
        self.model.eval()
        [self.log_acc[i].reset() for i in range(10)]
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        self.model.eval()
        data_iter = tqdm(self.data['test'],dynamic_ncols = True)
        for X, y, mask, index in data_iter:
            B:int = X.shape[0]
            T:int = X.shape[2]
            y_hat = self.model(X)
            y = y.to(self.device)

            loss = self.loss_func(y_hat, y)
            _, predicted_labels = torch.max(y_hat, 1)
            self.loss.update(loss.data.item(),B)
            for i, ratio in enumerate([(i+1)/10 for i in range(10)]):

                self.log_acc[i].update((predicted_labels == y.data)\
                                        .view(self.n_classes*B,-1)[:,int(math.ceil(T*ratio))-1].float().mean(), B)
            
            AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
            data_iter.set_description(
                f"[Epoch #{epoch}] "\
                f"AUC:{AUC:.3f}, " \
                f"LOSS:{self.loss.avg:.3f}, " 
            )
            test_logs:dict = {
            "test/LOSS":self.loss.avg,
            "test/AUC":AUC,
            }
            test_logs.update({f"test/ACC_{(i+1)/10}":self.log_acc[i].avg for i in range(10)})
            wandb.log(test_logs)

    def run(self) -> None:
        
        begin_epoch:int = 0
        new_exp:str = os.path.join(self.log_dir,'exp-')
        _, dirs, _ = next(os.walk(self.log_dir))
        dirs.sort()
        if len(dirs) == 0:

            exp_path = 'exp-0'

        else:

            exp_path = dirs[-1]

        last_exp_path = os.path.join(self.log_dir, exp_path,'checkpoint')
        if self.auto_resume and os.path.exists(last_exp_path):

            checkpoint = torch.load(last_exp_path)

            if checkpoint['epoch'] < self.max_epoch:

                logging.info('Loading model state from: '+last_exp_path+',Epoch: ' + checkpoint['epoch'])
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.step = checkpoint['step']
                begin_epoch = checkpoint['epoch']
                new_exp = new_exp + str(int(dirs[-1].split('-')[-1]))

            else:

                logging.info('No model checkpoint matched. Create new experiment.')

                self.log_dir = new_exp + str(int(exp_path.split('-')[-1]) + 1)
                os.mkdir(self.log_dir)
                

        else:

            self.log_dir = new_exp + str(int(exp_path.split('-')[-1]) + 1)
            os.mkdir(self.log_dir)
        
        for epoch in range(begin_epoch, self.max_epoch):

            self.train(epoch)

            if epoch % self.validate_per_epoch == 0:
                
                self.validate(epoch)

        self.save_log(new_exp)


    def save_log(self, exp_path:str) -> None:

        pass


    def adjust_learning_rate(self, epoch:int) -> float:
        '''
        Return adjusted learning rate.
        '''
        if args.optimizer == 'SGD' or args.optimizer == 'Adam' :

            if epoch < args.warm_up_epoch and args.auto_resume is False:

                lr = args.base_learning_rate * (epoch + 1) / args.warm_up_epoch

            else:

                lr = args.base_learning_rate * (
                        args.lr_decay_rate ** np.sum(epoch >= np.array(args.step)))
                
            for param_group in self.optimizer.param_groups:

                param_group['lr'] = lr

            return lr
        
        else:

            raise ValueError()
     

if __name__  == '__main__':

    wandb.login()
    args = parse_args()
    processor = Processor(args)
    processor.run()