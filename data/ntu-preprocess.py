'''
This file is used to preprocess ntu120 dataset. 

Usage: python ntu-preprocess.py --raw_data_dir raw_data_folder --target_dir ./

'''

import argparse
import os
from get_raw_skes_data import raw_preprocess
from get_raw_denoised_data import denoised_preprocess
from seq_transformation import padding_preprocess

def parse_args():

    # ===============================Project ============================

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--raw_data_dir',type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)

    parser.add_argument('--method', type=str, choices=['dynamic','padding'])
    parser.add_argument('--length', type=int, default=30)
 
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    raw_preprocess(args.raw_data_dir)
    denoised_preprocess(args.raw_data_dir)
    padding_preprocess(args.raw_data_dir, args.target_dir)