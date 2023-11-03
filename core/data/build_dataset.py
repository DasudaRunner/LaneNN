from .dataset import OpenLaneDataset

def build_dataset(dataset_kwargs):
    print(f"Dataset: {dataset_kwargs['type']}")
    return globals()[dataset_kwargs['type']](dataset_kwargs)