# MNIST Document Processing with PyTorch

A comprehensive neural network system for processing handwritten documents using the MNIST dataset as a foundation.

## Features

- **Multiple Model Architectures**: CNN and ResNet implementations
- **Complete Training Pipeline**: With TensorBoard logging and model checkpointing
- **Data Augmentation**: Built-in augmentation for better generalization
- **Evaluation Tools**: Confusion matrices, classification reports, and sample predictions
- **Easy Setup**: Automated requirement installation and directory creation

## Quick Start

### 1. Install Requirements

**Option A: Using the setup script (Recommended)**
```bash
python setup.py
```

**Option B: Using pip directly**
```bash
pip install -r requirements.txt
```

**Option C: Using batch/shell scripts**
- Windows: `install_requirements.bat`
- Unix/Linux/Mac: `./install_requirements.sh`

### 2. Train a Model

```bash
# Train CNN model for 10 epochs
python main.py --mode train --model cnn --epochs 10

# Train ResNet model for 15 epochs
python main.py --mode train --model resnet --epochs 15
```

### 3. Evaluate a Trained Model

```bash
# Evaluate the best model
python main.py --mode evaluate --model cnn
```

### 4. Make Predictions

```bash
# Run predictions on sample data
python main.py --mode predict --model cnn
```

## Project Structure

```
ComputerVision_NeuralNetwork/
├── models/                 # Neural network architectures
│   ├── cnn_models.py     # CNN and ResNet models
│   └── __init__.py
├── utils/                 # Data loading and utilities
│   ├── data_loader.py    # MNIST data loading
│   └── __init__.py
├── training/              # Training pipeline
│   ├── trainer.py        # Training loop and validation
│   └── __init__.py
├── inference/             # Prediction and evaluation
│   ├── predictor.py      # Model prediction utilities
│   └── __init__.py
├── configs/               # Configuration files
│   └── mnist_config.yaml
├── data/                  # Data storage (auto-created)
├── runs/                  # TensorBoard logs (auto-created)
├── main.py               # Main application
├── setup.py              # Setup script
├── requirements.txt       # Python dependencies
└── README.md
```

## Model Architectures

### 1. MNISTCNN
- Simple CNN with 3 convolutional layers
- Max pooling and dropout for regularization
- ~1.2M parameters
- Good for quick experimentation

### 2. MNISTResNet
- ResNet-inspired architecture with residual connections
- Batch normalization for stable training
- ~1.8M parameters
- Better performance on complex patterns

## Training Features

- **Automatic Data Splitting**: 80% train, 20% validation
- **Data Augmentation**: Random rotation and affine transformations
- **Learning Rate Scheduling**: Step decay for better convergence
- **Model Checkpointing**: Saves best model based on validation accuracy
- **TensorBoard Logging**: Track loss, accuracy, and learning rate
- **Progress Bars**: Real-time training progress with tqdm

## Evaluation Features

- **Classification Report**: Precision, recall, F1-score for each digit
- **Confusion Matrix**: Visual representation of predictions vs true labels
- **Sample Predictions**: Visualize model predictions with confidence scores
- **Batch Prediction**: Efficient prediction on multiple images

## Configuration

Edit `configs/mnist_config.yaml` to modify:
- Model architecture parameters
- Training hyperparameters
- Data augmentation settings
- Inference thresholds

## Expected Performance

With the default settings:
- **CNN Model**: ~98-99% test accuracy after 10 epochs
- **ResNet Model**: ~99%+ test accuracy after 10 epochs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in main.py
2. **Import Errors**: Run `python setup.py` to install missing packages
3. **Slow Training**: Use GPU if available, or reduce model complexity

### GPU Usage

The system automatically detects and uses GPU if available. To force CPU usage:
```python
device = torch.device('cpu')
```

## Extending the System

### Adding New Models
1. Create new model class in `models/cnn_models.py`
2. Add to model selection in `main.py`
3. Update configuration file

### Adding New Datasets
1. Create new data loader in `utils/data_loader.py`
2. Modify preprocessing for your data format
3. Update model input dimensions

### Adding New Evaluation Metrics
1. Add metric calculation in `inference/predictor.py`
2. Integrate with evaluation pipeline

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests!
