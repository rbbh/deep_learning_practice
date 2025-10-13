import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Preprocessor:
    def __init__(self, image_size=(32, 32), batch_size=64):
        self.image_size = image_size
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_data_loaders(self):
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
