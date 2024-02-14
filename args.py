import argparse
import os

def parse_args():

    # ===============================Project ============================

    parser = argparse.ArgumentParser(
        description='Arguments for ESG-GCN')
    
    parser.add_argument('--project_name',type=str, default='ESG-GCN')

    parser.add_argument('--dataset_name', type=str, default='ntu120',choices=['ntu120','ntu','ucla'])

    parser.add_argument('--split_type', type=str, default="CSet", choices=['CSet','CSub'])

    parser.add_argument('--device', type=str, default='mps')
    
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')

    parser.add_argument('--save_dir',type=str, default='checkpoints')

    parser.add_argument('--auto_resume',type=bool,default=True)

    parser.add_argument('--config_path', type=str, default=os.path.join('configs','ESRGCN-NTU.yaml'))

    parser.add_argument('--log_dir', type=str, default=os.path.join('logs'))

    parser.add_argument('--n_classes', type=int, default=120)

    parser.add_argument('--validate_per_epoch', type = int, default=1)

    # ===============================Data ================================

    parser.add_argument('--prefetch', default=4, type=int,
                        help="use for training duration per worker")

    parser.add_argument('--num_workers', default=14, type=int,
                        help="num_workers for dataloader")
    
    parser.add_argument('--random_rot', type=bool, default=True)

    parser.add_argument('--window_size', type = int, default=64)
    # ============================ Hyper parameters ========================================

    parser.add_argument('--base_learning_rate', type=float,
                        default=1e-3, help='learning rate.')

    parser.add_argument('--max_epoch', type=int, default=200)

    parser.add_argument('--batch_size', default=512, type=int,
                        help="use for training duration per worker")

    parser.add_argument('--val_batch_size', default = 512,
                        type=int, help="use for validation duration per worker")
    
    parser.add_argument('--label_smoothing', default=0, type=float)


    #============================Optimizer============================

    parser.add_argument('--nesterov', default=True, type=bool)

    parser.add_argument('--optimizer', default='Adam', type=str)

    parser.add_argument('--lr_decay_rate', default=0.1, type=float)

    parser.add_argument('--weight_decay', default=3e-4,type=float)

    parser.add_argument('--warm_up_epoch', default=5, type=int)

    parser.add_argument('--step', type=int, default=[50, 60], nargs='+', help='the epoch where optimizer reduce the learning rate')

    #==============================Debug config =======================


    return parser.parse_args()
