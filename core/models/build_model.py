from .LaneNN import LaneModelSeg, LaneModelCls
from utils.util import print_json

def build_model(model_kwargs):
    print(f"Model: {model_kwargs['type']}")
    print_json(model_kwargs['kwargs'])
    return globals()[model_kwargs['type']](**model_kwargs['kwargs'])