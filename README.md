# Self-supervised visual representation learning: SimCLR & Deep Clustering

This repository contains implementations of two self-supervised learning approaches; the SimCLR (Simple Framework for Contrastive Learning of Visual Representations) model for self-supervised learning on the CIFAR-10 dataset. The implementation includes data preparation, model definition, training, and visualization of embeddings using t-SNE and the DeepCluster algorithm for unsupervised learning of visual features using the CIFAR-10 dataset. The code is organized into modular Python scripts to facilitate understanding and reuse.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Directory Structure

SimCLR/
- ├── `data.py`: Handles data loading and transformations for the CIFAR-10 dataset.
- ├── `losses.py`: Contains the implementation of the contrastive loss function (NT-Xent loss) used in SimCLR.
- ├── `main.py`: The main script to run the training process.
- ├── `models.py`: Defines the SimCLR model architecture, including the encoder and projection head.
- ├── `train.py`: Contains the training loop for the SimCLR model.
- ├── `utils.py`: Utility functions used throughout the project.
- └── `vis.py`: Provides functions for visualizing embeddings using t-SNE.

DeepCluster/
- ├── `models.py`: Model definitions
- ├── `train.py`: Training functions
- ├── `utils.py`: Utility functions
- └── `main.py`: Main script to run the project

`README.md`: Provides an overview of the project, including installation, usage, and details about the implementation.

`requirements.txt`: Lists the dependencies and packages required to run the project.


## Prerequisites

- Python 3.6 or later
- `torch` and `torchvision` packages
- Install the packages in the requirements.txt file

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pmensah28/self-supervised-learning.git

   ```
2. Change directory to your prefered method (SimCLR or DeepCluster) and run the `main.py` file.

