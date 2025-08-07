import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class MNISTPredictor:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict_single(self, image):
        """Predict on a single image"""
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image.unsqueeze(0))
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1)
            
        return prediction.item(), probabilities.squeeze().cpu().numpy()
    
    def predict_batch(self, images):
        """Predict on a batch of images"""
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def evaluate_model(self, test_loader):
        """Evaluate model on test set"""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = [str(i) for i in range(10)]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    def plot_sample_predictions(self, test_loader, num_samples=16):
        """Plot sample predictions with confidence scores"""
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        
        predictions, probabilities = self.predict_batch(images)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_samples):
            img = images[i].squeeze().numpy()
            true_label = labels[i].item()
            pred_label = predictions[i]
            confidence = probabilities[i][pred_label]
            
            axes[i].imshow(img, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}', 
                            color=color)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show() 