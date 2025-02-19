from torchvision import transforms
from typing import Tuple


class Config:
    epochs: int = 4

    sample_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    layers_per_block: int = 2
    block_out_channels: Tuple = (128, 128, 256, 256)

    down_block_types: Tuple = (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )

    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ]
    )
