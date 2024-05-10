import torch


class Config:
    def __init__(
        self,
        lr=3e-4,
        num_epochs=300,
        img_size=64,
        gray_scale=False,
        channels=3,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        # TODO get number of classes of the dataset
        num_classes=10,
        out_dir="out",
        time_dim=256,
        batch_size=14,
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

        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.lr = lr
        self.time_dim = time_dim


tiny_config = Config()

base_config = Config()
