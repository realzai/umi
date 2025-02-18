from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset
from umi.config import Config
from umi.models.unet import create_model
from umi.datasets import CIFAR10Dataset
from diffusers import DDPMPipeline, DDPMScheduler

if __name__ == "__main__":
    config = Config()

    model = create_model(config)
    dataset = load_dataset("cifar10", split="train")
    dataset = CIFAR10Dataset(dataset, transform=config.transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(config.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress_bar):
            clean_images = batch["image"]
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=clean_images.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            model_output = model(noisy_images, timesteps).sample
            loss = torch.nn.functional.mse_loss(model_output, noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

    model.save_pretrained("ddpm-cifar10")
    noise_scheduler.save_pretrained("ddpm-cifar10")

    # Push to huggingface
    # model.push_to_hub("zaibutcooler/umi")
    # noise_scheduler.push_to_hub("zaibutcooler/umi")
