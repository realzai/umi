from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["img"]
        if self.transform:
            img = self.transform(img)
        return {"image": img}
