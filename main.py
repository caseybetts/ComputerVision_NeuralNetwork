import torch
import torch.nn as nn
from models.cnn_models import MNISTCNN, MNISTResNet
from utils.data_loader import MNISTDataLoader
from training.trainer import MNISTTrainer
from inference.predictor import MNISTPredictor
from sklearn.metrics import classification_report  # Add this import
import argparse

def main():
    parser = argparse.ArgumentParser(description='MNIST Document Processing with PyTorch')
    parser.add_argument('--model', choices=['cnn', 'resnet'], default='cnn', 
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], 
                       default='train', help='Mode to run')
    parser.add_argument('--no_plot', action='store_true', help='Skip plotting to avoid hanging')
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
        print(f"\nStarting training for {args.epochs} epochs...")
        trainer = MNISTTrainer(model, train_loader, val_loader, device)
        train_losses, val_losses, train_accs, val_accs = trainer.train(epochs=args.epochs)
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETED!")
        print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
        print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
        print(f"Best Model saved as: best_model.pth")
        print(f"{'='*50}")
        
        # Plot training history (optional)
        if not args.no_plot:
            print("\nGenerating training plots...")
            trainer.plot_training_history()
            print("Plot window opened. Close it to continue or press Ctrl+C to exit.")
        else:
            print("\nSkipping plots (--no_plot flag used)")
            print("Training completed successfully!")
        
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