import torch
import torch.nn as nn
from models.cnn_models import EMNISTCNN, EMNISTResNet, MNISTCNN, MNISTResNet
from utils.data_loader import EMNISTDataLoader, MNISTDataLoader
from training.trainer import MNISTTrainer
from inference.predictor import MNISTPredictor
from inference.document_reader import DocumentReader
from sklearn.metrics import classification_report
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='EMNIST Document Processing with PyTorch')
    parser.add_argument('--model', choices=['cnn', 'resnet'], default='cnn', 
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict', 'read_document'], 
                       default='train', help='Mode to run')
    parser.add_argument('--no_plot', action='store_true', help='Skip plotting to avoid hanging')
    parser.add_argument('--document_path', type=str, help='Path to document image for reading')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--dataset', choices=['letters', 'digits', 'balanced'], 
                       default='letters', help='Dataset to use (letters, digits, or balanced)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.mode == 'read_document':
        # Document reading mode
        if not args.document_path:
            print("Error: --document_path is required for read_document mode")
            return
        
        if not os.path.exists(args.document_path):
            print(f"Error: Document file not found: {args.document_path}")
            return
        
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found: {args.model_path}")
            print("Please train a model first using: python main.py --mode train --dataset letters")
            return
        
        print(f"Reading document: {args.document_path}")
        reader = DocumentReader(args.model_path, device, dataset_type=args.dataset)
        
        # Read document
        result = reader.read_document(args.document_path, visualize=True)
        
        print("\n" + "="*50)
        print("DOCUMENT READING RESULTS")
        print("="*50)
        print(f"Dataset type: {args.dataset}")
        print(f"Number of text regions: {result['num_regions']}")
        print(f"Total characters detected: {result['num_characters']}")
        print("\nExtracted text:")
        print("-"*30)
        print(result['text'])
        print("-"*30)
        
        # Save results
        output_file = args.document_path.replace('.jpg', '_extracted.txt').replace('.png', '_extracted.txt')
        with open(output_file, 'w') as f:
            f.write(result['text'])
        print(f"\nResults saved to: {output_file}")
        
    else:
        # Training/evaluation modes
        # Load data
        data_loader = EMNISTDataLoader(batch_size=args.batch_size, dataset=args.dataset)
        train_loader, val_loader, test_loader = data_loader.get_data_loaders()
        
        # Get number of classes
        num_classes = data_loader.get_num_classes()
        class_names = data_loader.get_class_names()
        
        # Create model
        if args.model == 'cnn':
            model = EMNISTCNN(num_classes=num_classes)
        else:
            model = EMNISTResNet(num_classes=num_classes)
        
        print(f'Model: {args.model.upper()}')
        print(f'Dataset: {args.dataset}')
        print(f'Number of classes: {num_classes}')
        print(f'Classes: {class_names}')
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
            print(classification_report(labels, predictions, target_names=class_names))
            
            # Plot confusion matrix
            predictor.plot_confusion_matrix(labels, predictions, class_names)
            
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
                true_label = class_names[labels[i].item()]
                pred_label = class_names[predictions[i]]
                confidence = probabilities[i][predictions[i]]
                print(f'Image {i+1}: True={true_label}, Pred={pred_label}, '
                      f'Confidence={confidence:.3f}')

if __name__ == '__main__':
    main() 