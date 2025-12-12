# MNIST Digit Classification with Baseline CNN

A PyTorch implementation of a Convolutional Neural Network for handwritten digit recognition using the MNIST dataset.

## Project Overview

This project implements a compact CNN architecture to classify handwritten digits (0-9) from the MNIST dataset. The model is designed to be lightweight yet effective, achieving good accuracy with minimal parameters.

## Architecture Details

### Model Architecture
```
Net(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (fc1): Linear(in_features=288, out_features=35, bias=True)
  (fc2): Linear(in_features=35, out_features=10, bias=True)
)
```

### Layer-by-Layer Breakdown

| Layer | Input Size | Output Size | Parameters | Description |
|-------|------------|-------------|------------|-------------|
| Conv1 | 1×28×28 | 16×28×28 | 160 | 3×3 conv, padding=1, ReLU activation |
| MaxPool1 | 16×28×28 | 16×14×14 | 0 | 2×2 max pooling |
| Conv2 | 16×14×14 | 32×14×14 | 4,640 | 3×3 conv, padding=1, ReLU activation |
| MaxPool2 | 32×14×14 | 32×7×7 | 0 | 2×2 max pooling |
| Conv3 | 32×7×7 | 32×7×7 | 9,248 | 3×3 conv, padding=1, ReLU activation |
| MaxPool3 | 32×7×7 | 32×3×3 | 0 | 2×2 max pooling |
| Flatten | 32×3×3 | 288 | 0 | Reshape to 1D |
| FC1 | 288 | 35 | 10,115 | Fully connected, ReLU activation |
| FC2 | 35 | 10 | 360 | Output layer |

**Total Parameters:** 24,523

### Key Design Choices

1. **Padding Strategy**: All convolutional layers use `padding=1` to preserve spatial dimensions
2. **Progressive Channels**: 1 → 16 → 32 → 32 channels for feature extraction
3. **Compact FC Layers**: Small fully connected layers (35 neurons) to minimize parameters
4. **Activation Functions**: ReLU for hidden layers, LogSoftmax for output

## Data Preprocessing

### Training Transforms
- **RandomApply CenterCrop**: 22×22 crop applied with 10% probability
- **Resize**: Restore to 28×28 pixels
- **RandomRotation**: ±15 degrees rotation with zero fill
- **Normalization**: Mean=0.1307, Std=0.3081 (MNIST statistics)

### Test Transforms
- **ToTensor**: Convert PIL Image to tensor
- **Normalization**: Same statistics as training

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | Adam | Learning rate: 0.01, Weight decay: 1e-4 |
| Scheduler | StepLR | Step size: 15, Gamma: 0.1 |
| Loss Function | NLLLoss | Negative Log Likelihood Loss |
| Batch Size | 128 | Training batch size |
| Epochs | 2 | Number of training epochs |

### Weight Initialization
- **Kaiming Normal**: Applied to Conv2d and Linear layers
- **Bias Initialization**: Set to zero

## Requirements

```
torch
torchvision
torchsummary
matplotlib
tqdm
numpy
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install torch torchvision torchsummary matplotlib tqdm
```

## Usage

### Training the Model
```python
python era_session_4.py
```

### Model Summary
The code includes model summary generation showing:
- Layer-wise output shapes
- Parameter count per layer
- Total trainable parameters

## Results Visualization

The project generates comprehensive training visualizations:
- Training Loss curve
- Training Accuracy curve  
- Test Loss curve
- Test Accuracy curve

## Model Performance

The model tracks:
- **Training Metrics**: Loss and accuracy per epoch
- **Validation Metrics**: Test loss and accuracy
- **Real-time Progress**: Training progress with tqdm bars

## File Structure

```
project/
├── era_session_4.py          # Main training script
├── README.md                 # This file
└── data/                     # MNIST dataset (auto-downloaded)
```

## Key Features

1. **Efficient Architecture**: Minimal parameters while maintaining performance
2. **Data Augmentation**: Random crops and rotations for better generalization
3. **Progressive Pooling**: Gradual spatial dimension reduction (28→14→7→3)
4. **Comprehensive Logging**: Detailed training progress and metrics
5. **GPU Support**: Automatic CUDA detection and usage

## Architecture Philosophy

This CNN follows modern design principles:
- **Feature Extraction**: Three convolutional layers with increasing depth
- **Spatial Reduction**: Strategic max pooling to reduce computational load
- **Classification Head**: Compact fully connected layers for final prediction
- **Regularization**: Weight decay and data augmentation to prevent overfitting

## Future Improvements

Potential enhancements:
- Batch normalization layers
- Dropout for additional regularization
- Deeper architecture exploration
- Advanced data augmentation techniques
- Model ensemble methods

## Training Logs

### Epoch 1
```
Train: Loss=0.3083 Batch_id=468 Accuracy=91.50: 100%|██████████| 469/469 [00:45<00:00, 10.30it/s]
Test set: Average loss: 0.0009, Accuracy: 57784/60000 (96.31%)
```

### Epoch 2
```
Train: Loss=0.1965 Batch_id=468 Accuracy=96.75: 100%|██████████| 469/469 [00:46<00:00, 10.00it/s]
Test set: Average loss: 0.0006, Accuracy: 58569/60000 (97.61%)
```

### Performance Summary
- **Final Training Accuracy**: 96.75%
- **Final Test Accuracy**: 97.61%
- **Training Speed**: ~10 batches/second
- **Total Training Time**: ~1.5 minutes (2 epochs)
