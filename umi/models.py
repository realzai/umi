import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin


class ConditonalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class DiffusionModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self) -> None:
        super().__init__()
