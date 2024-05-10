import huggingface_hub
import torch
from tqdm import tqdm
from torch import nn, optim

from .model import ConditonalUNet, Diffusion
from .config import Config
from .dataset import TrainData
from .utils import save_images, display_images


class Umi:
    def __init__(self, config: Config) -> None:
        self.device = config.deivce
        self.model = ConditonalUNet(config)
        self.diffuser = Diffusion()
        self.config = Config()

    def train(self, dataset: TrainData):
        config = self.config
        device = config.deivce
        optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        loss_fn = nn.MSELoss()
        dataloader = dataset.get_data_loader()

        for epoch in range(config.num_epochs):
            for i, (image, label) in enumerate(dataloader):
                pass

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = self.sample(n=len(labels), labels=labels)

    def generate(self):
        pass

    def sample(self, n, labels, cfg_scale=3):
        print(f"Sampling {n} new images....")
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = self.model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = self.model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
        self.model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    @torch.no_grad()
    def evaluate(self, dataset: TrainData):
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
