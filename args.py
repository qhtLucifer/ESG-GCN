import argparse
import os

def parse_args():

    # ===============================Project ============================

    parser = argparse.ArgumentParser(
        description='Arguments for ESG-GCN')
    
    parser.add_argument('--project_name',type=str, default='ESG-GCN')

    parser.add_argument('--dataset_name', type=str, default='ntu120',choices=['ntu120','ntu','ucla'])

    parser.add_argument('--device', type=str, default='mps')
    
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')

    parser.add_argument('--save_dir',type=str, default='checkpoints')

    parser.add_argument('--auto_resume',type=bool,default=True)

    parser.add_argument('--config_path', type=str, default=os.path.join('configs','ESRGCN-NTU.yaml'))

    # ===============================Data ================================

    parser.add_argument('--prefetch', default=4, type=int,
                        help="use for training duration per worker")

    parser.add_argument('--num_workers', default=14, type=int,
                        help="num_workers for dataloader")

    # ============================ Hyper parameters ========================================

    parser.add_argument('--learning_rate', type=float,
                        default=1e-3, help='learning rate.')

    parser.add_argument('--max_epochs', type=int, default=200)

    parser.add_argument('--batch_size', default=512, type=int,
                        help="use for training duration per worker")

    parser.add_argument('--val_batch_size', default = 512,
                        type=int, help="use for validation duration per worker")

    #==============================Debug config =======================


    return parser, parser.parse_args(args=[])
