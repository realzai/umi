from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms

class TrainData(Dataset):
    def __init__(self):
        self.images = None
        self.labels = None
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return []
