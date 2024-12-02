import math
import numpy as np
import pandas as pd

from utill import *
import torch.optim as optim
class Alg:
    def __init__(self,name, config,device):
        self.name = name
        self.config = config
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.training_mode = config.training_mode

    def set_algo(self,model_dir,start_epoch):
        pass
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, batch):
        return self.model(batch)
