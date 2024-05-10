import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from .config import Config
from .components import DoubleConv,Down,SelfAttention,Up
import torch.functional as F

class Diffusion:
    def __init__(self) -> None:
        pass
    
class ConditonalUNet(nn.Module,PyTorchModelHubMixin):
    def __init__(self,config:Config,num_classes=None):
        super().__init__()
        self.time_dim = config.time_dim
        self.device = config.deivce
        
        self.in_channel = DoubleConv(config.channels,64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.out_channel = nn.Conv2d(config.channels,kernel_size=1)
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, config.time_dim)
        

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self,x,t,y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)
        
        x1 = self.in_channel(x)
        x2 = self.down1(x1,t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2,t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3,t)
        x4 = self.sa3(x4)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4,x3,t)
        x = self.sa4(x)
        x = self.up2(x,x2,t)
        x = self.sa5(x)
        x = self.up3(x,x1,t)
        x = self.sa6(x)
        output  = self.out_channel(x)
        return output
    
