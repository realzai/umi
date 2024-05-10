from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from.config import Config


class TrainData(Dataset):
    def __init__(self, config: Config, shuffle=True):
        self.loaded_data = load_dataset("zaibutcooler/beauty")
        self.images = self.loaded_data["train"]["image"]
        self.labels = self.loaded_data["train"]["label"]
        self.transform = transforms.Compose([
            transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.batch_size = config.batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img = self.transform(img)
        return img, label

    def get_data_loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)