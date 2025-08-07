import torch
import torch.nn as nn
from models.cnn_models import MNISTCNN, MNISTResNet
from utils.data_loader import MNISTDataLoader
from training.trainer import MNISTTrainer
from inference.predictor import MNISTPredictor
import argparse

def main():
    parser = argparse.ArgumentParser(description='MNIST Document Processing with PyTorch')
    parser.add_argument('--model', choices=['cnn', 'resnet'], default='cnn', 
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], 
                       default='train', help='Mode to run')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    data_loader = MNISTDataLoader(batch_size=args.batch_size)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Create model
    if args.model == 'cnn':
        model = MNISTCNN(num_classes=10)
    else:
        model = MNISTResNet(num_classes=10)
    
    print(f'Model: {args.model.upper()}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    if args.mode == 'train':
        # Training
        trainer = MNISTTrainer(model, train_loader, val_loader, device)
        train_losses, val_losses, train_accs, val_accs = trainer.train(epochs=args.epochs)
        
        # Plot training history
        trainer.plot_training_history()
        
    elif args.mode == 'evaluate':
        # Load best model
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        
        # Evaluate
        predictor = MNISTPredictor(model, device)
        predictions, labels, probabilities = predictor.evaluate_model(test_loader)
        
        # Print classification report
        print('\nClassification Report:')
        print(classification_report(labels, predictions))
        
        # Plot confusion matrix
        predictor.plot_confusion_matrix(labels, predictions)
        
        # Plot sample predictions
        predictor.plot_sample_predictions(test_loader)
        
    elif args.mode == 'predict':
        # Load best model
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        
        # Interactive prediction
        predictor = MNISTPredictor(model, device)
        
        # Get a sample from test set
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        
        # Make predictions
        predictions, probabilities = predictor.predict_batch(images[:5])
        
        print('\nSample Predictions:')
        for i in range(5):
            print(f'Image {i+1}: True={labels[i].item()}, Pred={predictions[i]}, '
                  f'Confidence={probabilities[i][predictions[i]]:.3f}')

if __name__ == '__main__':
    main() 