import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

class MNISTDataLoader:
    def __init__(self, batch_size=64, train_split=0.8):
        self.batch_size = batch_size
        self.train_split = train_split
        
        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def get_data_loaders(self):
        """Get train, validation, and test data loaders"""
        
        # Load full training dataset
        full_train_dataset = datasets.MNIST(
            root='./data', 
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
        test_dataset = datasets.MNIST(
            root='./data', 
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
    
    def visualize_batch(self, data_loader, num_samples=8):
        """Visualize a batch of images"""
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            img = images[i].squeeze().numpy()
            label = labels[i].item()
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return images, labels 