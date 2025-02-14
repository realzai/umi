from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset
from umi.config import Config
from umi.models.unet import create_model
from umi.datasets import CIFAR10Dataset
from diffusers import DDPMPipeline, DDPMScheduler

if __name__ == '__main__':
    config = Config()

    model = create_model(config)
    dataset = load_dataset("cifar10", split="train")
    dataset = CIFAR10Dataset(dataset, transform=config.transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)