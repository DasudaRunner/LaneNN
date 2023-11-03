from .openlane_trainer import OpenLaneTrainer

def build_trainer(C):
    print(f"Trainer: {C['common']['trainer']}")
    return globals()[C['common']['trainer']](C)