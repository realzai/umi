import torch


class Config:
    def __init__(
        self,
        img_size=256,
        gray_scale=False,
        channels=3,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        out_dir = "out",
        time_dim = 256,
        batch_size=32
    ):
        self.img_size: int = img_size
        self.gray_scale: bool = gray_scale
        self.channels: int = channels

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.batch_size = batch_size
        self.deivce = "cuda" if torch.cuda.is_available() else "cpu"
        self.out_dir = out_dir
        
        self.num_epochs = 40
        self.lr = 0.001
        self.time_dim = time_dim


tiny_config = Config()

base_config = Config()