import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

class EMNISTDataLoader:
    def __init__(self, batch_size=64, train_split=0.8, dataset='letters'):
        self.batch_size = batch_size
        self.train_split = train_split
        self.dataset = dataset  # 'letters', 'digits', or 'balanced'
        
        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # EMNIST normalization
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def get_data_loaders(self):
        """Get train, validation, and test data loaders"""
        
        # Load full training dataset
        full_train_dataset = datasets.EMNIST(
            root='./data', 
            split=self.dataset,
            train=True, 
            download=True, 
            transform=self.train_transform
        )
        
        # Split into train and validation
        train_size = int(self.train_split * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )
        
        # Load test dataset
        test_dataset = datasets.EMNIST(
            root='./data', 
            split=self.dataset,
            train=False, 
            download=True, 
            transform=self.test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_names(self):
        """Get class names for the selected dataset"""
        if self.dataset == 'letters':
            return [chr(i) for i in range(65, 91)]  # A-Z
        elif self.dataset == 'digits':
            return [str(i) for i in range(10)]  # 0-9
        elif self.dataset == 'balanced':
            return [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]  # 0-9, A-Z
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    def get_num_classes(self):
        """Get number of classes for the selected dataset"""
        if self.dataset == 'letters':
            return 26  # A-Z
        elif self.dataset == 'digits':
            return 10  # 0-9
        elif self.dataset == 'balanced':
            return 36  # 0-9 + A-Z
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    def visualize_batch(self, data_loader, num_samples=8):
        """Visualize a batch of images"""
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        class_names = self.get_class_names()
        
        for i in range(num_samples):
            img = images[i].squeeze().numpy()
            label = labels[i].item()
            label_name = class_names[label] if label < len(class_names) else '?'
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {label_name}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return images, labels

# Keep the old MNISTDataLoader for backward compatibility
class MNISTDataLoader(EMNISTDataLoader):
    def __init__(self, batch_size=64, train_split=0.8):
        super().__init__(batch_size, train_split, dataset='digits') 