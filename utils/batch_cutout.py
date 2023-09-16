import torch
from utils.cutout import Cutout

class batch_Cutout(torch.nn.Module):
    def __init__(self, n_holes, length):
        super().__init__()
        self.n_holes = n_holes
        self.length = length

    def forward(self, img):
        return Cutout(img, self.n_holes, self.length)
