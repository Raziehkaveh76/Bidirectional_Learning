# Bidirectional Neural Networks for MNIST Classification

This repository contains implementations of Bidirectional Neural Networks (ANN and CNN) for MNIST classification, with a focus on robustness against adversarial attacks.

## Project Structure

```
├── csv/                           # Training and evaluation metrics
│   ├── cnn_mnist_nobias_backprop/
│   ├── cnn_mnist_nobias_biprop/
│   └── cnn_mnist_nobias_halfbiprop/
├── Weights/                       # Model weights
│   ├── nobias_backprop/
│   ├── nobias_biprop/
│   └── nobias_halfbiprop/
├── visualization/                 # Static visualizations
│   ├── robustness_analysis/      # Model robustness metrics
│   ├── ANN_perfomance_visual/    # ANN performance plots
│   ├── CNN_2D_visualization/     # 2D CNN visualizations
│   └── ANN_2D_visualization/     # 2D ANN visualizations
├── visualization_3d/             # 3D visualizations
│   └── animations/              # Animated GIFs of model behavior
└── Scripts                      # Analysis and visualization scripts
    ├── ANN_animation.py        # 3D PCA visualization of ANN behavior
    ├── CNN_performance_visual.py # CNN performance plotting
    ├── ANN_robustness_analysis.py # Robustness analysis
    ├── ANN_2D_visual.py        # 2D ANN visualizations
    ├── CNN_2D_visual_sharePCA.py # 2D CNN visualizations with shared PCA
    └── utils/                  # Utility functions
        ├── utils_csv.py       # CSV handling utilities
        └── utils_tf2.py       # TensorFlow 2.x utilities
```

## Experiment Types

1. **Backpropagation (BP)**
   - Standard neural network training
   - Results in `*_backprop` directories

2. **Bidirectional Learning (BL)**
   - Novel training approach with bidirectional learning
   - Results in `*_biprop` directories

3. **Half Bidirectional Learning (BL->BP)**
   - Hybrid approach combining BP and BL
   - Results in `*_halfbiprop` directories

## Running Different Configurations

Both ANN and CNN models can be run with different configurations using the following commands:

```bash
# For CNN model
python mnist_cnn_gan_2conv.py <config_number>

# For ANN model
python mnist_nn_nobias.py <config_number>
```

Configuration numbers (same for both models):
- `0`: Backpropagation with bias
- `1`: Bidirectional propagation with bias
- `2`: Half bidirectional propagation with bias
- `3`: Backpropagation without bias
- `4`: Bidirectional propagation without bias
- `5`: Half bidirectional propagation without bias

Examples:
```bash
# Run CNN with backpropagation without bias
python mnist_cnn_gan_2conv.py 3

# Run ANN with bidirectional propagation with bias
python mnist_nn_nobias.py 1
```

The results will be saved in their respective directories under `csv/` and `Weights/`.

## Visualizations

### Performance Visualizations
- Training progress for each model:
  - Clean Test Data accuracy (green)
  - Noisy Test Data accuracy (blue)
  - FGSM Test Data accuracy (red)

### 2D Visualizations
- PCA-based visualizations for both ANN and CNN models
- Shared PCA space analysis for CNN
- Feature space analysis

### 3D PCA Visualizations
- Location: `visualization_3d/animations/`
- Animated GIFs showing:
  - Data distribution in 3D PCA space
  - Class separation (digits 2,3,4)
  - Model behavior under different conditions:
    - Clean data
    - Noisy data
    - FGSM attacks

### Robustness Analysis
- Comprehensive model robustness metrics
- Adversarial attack impact analysis
- Performance comparison across models

## Data Organization

### CSV Files
- Training metrics and evaluation results
- Separate directories for each model configuration
- Includes accuracy, sigmoid, and softmax data

### Weights
- Stored model parameters
- No-bias versions of each configuration
- Used for model analysis and visualization

## Requirements

Required Python packages:
- numpy==1.23.5
- matplotlib==3.7.1
- scikit-learn==1.2.2
- tensorflow==2.12.0
- imageio==2.31.1
- scipy==1.10.1
- cleverhans==4.0.0
