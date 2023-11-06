import numpy as np
import matplotlib.pyplot as plt
from utils.misc import load_yaml
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse

from core.trainer.build_trainer import build_trainer

parser = argparse.ArgumentParser(description='OD training')
parser.add_argument('--config', type=str, required=True, help='path to config file')
parser.add_argument('--tag', type=str, required=True, help='exp name')
parser.add_argument('--load_path', type=str, help='load mode path')
parser.add_argument('--eval', action='store_true')

def main():
    args = parser.parse_args()
    C = load_yaml(args.config)
    C['tag'] = args.tag    
    C['config_file'] = args.config
    if args.eval:
        C['eval'] = True
        C['load_path'] = args.load_path
    trainer = build_trainer(C)
    trainer.eval()
    
if __name__ == '__main__':
    main()