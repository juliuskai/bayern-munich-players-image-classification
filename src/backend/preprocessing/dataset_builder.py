from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def get_dataloaders(data_dir, batch_size=32, val_split=0.2):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # load full dataset with class labels
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    # extract labels for stratified split!!!
    targets = full_dataset.targets  # -> list of class indices

    # stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(full_dataset.samples, targets))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes, np.array(targets)[train_idx]
