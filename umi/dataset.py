from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from .config import Config


class TrainData(Dataset):
    def __init__(self, config: Config):
        self.loaded_data = load_dataset("zaibutcooler/beauty")
        self.images = self.loaded_data["train"]["images"]
        self.labels = self.loaded_data["train"]["labels"]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(config.img_size),
                # TODO double check the code
                transforms.Grayscale() if config.gray_scale else None,
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img = self.transform(img)
        return img, label
