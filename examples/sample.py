from umi import Umi

# from umi.dataset import TrainData
from umi.config import base_config
from umi.utils import display_images, save_images
from umi.model import ConditonalUNet

umi = Umi(config=base_config)

umi.load_pretrained()

images = umi.sample()

# save_images()
