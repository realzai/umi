import huggingface_hub
import torch

from .model import ConditonalUNet,Diffusion
from .config import Config
from .dataset import TrainData
from .utils import save_images,display_images
from torch import nn,optim


class Umi:
    def __init__(self,config:Config) -> None:
        self.device = config.deivce
        self.model = ConditonalUNet(config)
        self.diffuser = Diffusion()
        self.config = Config()
    
    def generate(self):
        pass
    
    def sample(self):
        pass
    
    def train(self,dataset:TrainData):
        config = self.config
        optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        loss_fn = nn.MSELoss()
        dataloader = dataset.get_data_loader()

        for epoch in range(config.num_epochs):
            for i,(img,label) in enumerate(dataloader):
                pass
            
        
        
    @torch.no_grad()
    def evaluate(self,dataset:TrainData):
        pass
    
    
    def fine_tune(self):
        pass
    
    def save_pretrained(self, name="umi"):
        self.model.save_pretrained(name)
        self.model.push_to_hub(name)
        print("Successfully saved the pretrainied")

    def load_pretrained(self, url="zaibutcooler/umi"):
        self.model = self.gpt.from_pretrained(url)
        print("Successfully loaded the pretrained")

    def huggingface_login(self, token):
        assert token is not None
        huggingface_hub.login(token=token)
        print("Logged in successfully")