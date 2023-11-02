from .base_trainer import MultitaskContrastTrainer


def build_trainer(C):
    dist_print(f"Trainer: {C.config['common']['trainer']}")
    return globals()[C.config['common']['trainer']](C)