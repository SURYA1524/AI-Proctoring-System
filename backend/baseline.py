import numpy as np

class Baseline:
    def __init__(self):
        self.eye = []
        self.head = []

    def update(self, eye_dev, head_dev):
        self.eye.append(eye_dev)
        self.head.append(head_dev)

    def get_baseline(self):
        return {
            "eye": np.mean(self.eye) if self.eye else 0,
            "head": np.mean(self.head) if self.head else 0
        }