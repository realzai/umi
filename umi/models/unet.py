from umi.config import Config
from diffusers import UNet2DModel


def create_model(config: Config):
    return UNet2DModel(
        sample_size=config.sample_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types,
    )
