from .dataset import OpenLaneDataset
from utils.util import print_json

def build_dataset(dataset_kwargs):
    print(f"Dataset: {dataset_kwargs['type']}")
    print_json(dataset_kwargs['kwargs'])
    dataset = globals()[dataset_kwargs['type']](dataset_kwargs['kwargs'])
    print(f'Dataset size: {len(dataset)}')
    return dataset