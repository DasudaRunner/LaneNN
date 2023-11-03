import os
import sys
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=20):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0

    def empty(self):
        return len(self.history) == 0

    def update(self, val, weight=1):
        self.history.append(val)
        if self.length > 0 and len(self.history) > self.length:
            del self.history[0]

        self.val = val
        self.avg = np.mean(self.history)