# CIFAR-10 Image Classification with Custom Neural Network Architecture

A deep learning project implementing a custom neural network architecture for CIFAR-10 image classification, achieving **86.84% test accuracy** through iterative improvements and optimisation techniques.

## 📋 Project Overview

This project demonstrates the development and optimisation of a neural network architecture called **RashadNet** for classifying images in the CIFAR-10 dataset. The architecture features a design with parallel convolutional paths and dynamic feature weighting, combined with modern training techniques to achieve strong performance.

## 🎯 Results

| Model Version | Key Features | Test Accuracy |
|--------------|--------------|---------------|
| Baseline | 3 blocks, basic architecture | 42.10% |
| First Wave | Data augmentation, batch norm, dropout, 6 blocks | 56.14% |
| **Final Model** | **Label smoothing, cosine annealing, MaxPool, 10 blocks** | **86.84%** |

## 🔧 Technologies Used

- **Python 3.11**
- **PyTorch**: Deep learning framework
- **torchvision**: Dataset loading and image transformations
- **matplotlib**: Visualisation
- **CUDA**: GPU acceleration

## 📊 Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) contains:
- 60,000 32×32 colour images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images
- 10,000 test images

## 🚀 Key Features & Techniques

### Data Augmentation
- Random horizontal flips
- Random cropping with padding
- Normalisation to [-1, 1] range

### Regularisation
- Batch normalisation after convolutions
- Dropout (p=0.3) in output block
- Label smoothing (0.1)

### Training Optimisation
- Adam optimiser
- Cosine annealing learning rate scheduler
- Cross-entropy loss with label smoothing

### Architecture Enhancements
- Deeper network (10 intermediate blocks)
- Wider network (64 base channels)
- MaxPool downsampling for feature preservation

## 📁 Project Structure

```
CIFAR-10 Image Classification/
├── html_export/
│   └── CIFAR10_image_classification.html     # Notebook exported as HTML
├── notebook/
│   └── CIFAR10_image_classification.ipynb    # Main Jupyter notebook
└── README.md                                   # Project documentation
```

## 🔬 Methodology

The project follows a structured approach:

1. **Dataset Preparation**: Loading and preprocessing CIFAR-10 with PyTorch DataLoaders
2. **Basic Architecture**: Implementing the custom RashadNet with intermediate blocks
3. **Training & Testing**: Establishing baseline performance and evaluation metrics
4. **Iterative Improvements**: Systematic enhancements through two major update waves

## 🙏 Acknowledgements

- CIFAR-10 dataset creators
- PyTorch community and documentation
- Inspiration from modern CNN architectures (ResNet, DenseNet, etc.)