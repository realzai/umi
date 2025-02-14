from torchvision import transforms

class Config:
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ])