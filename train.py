import numpy as np
import matplotlib.pyplot as plt
from utils.misc import load_yaml
import os
import argparse

parser = argparse.ArgumentParser(description='OD training')
parser.add_argument('--config', type=str, required=True, help='path to config file')
parser.add_argument('--tag', type=str, required=True, help='exp name')

def main():
    args = parser.parse_args()
    C = load_yaml(args.config)
    C['tag'] = args.tag
    trainer = build_trainer(C)
    trainer.run()
    
if __name__ == '__main__':
    main()