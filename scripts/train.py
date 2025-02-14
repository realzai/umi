from umi.config import Config
from umi.models.unet import create_model

if __name__ == '__main__':
    model_config = Config()

    model = create_model(model_config)
