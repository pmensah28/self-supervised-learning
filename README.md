# Self-supervised visual representation learning: SimCLR & Deep Clustering

This repository contains implementations of two self-supervised learning approavhes; the SimCLR (Simple Framework for Contrastive Learning of Visual Representations) model for self-supervised learning on the CIFAR-10 dataset. The implementation includes data preparation, model definition, training, and visualization of embeddings using t-SNE and the DeepCluster algorithm for unsupervised learning of visual features using the CIFAR-10 dataset. The code is organized into modular Python scripts to facilitate understanding and reuse.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [License](#license)

## Directory Structure

- `data.py`: Handles data loading and transformations for the CIFAR-10 dataset.
- `losses.py`: Contains the implementation of the contrastive loss function (NT-Xent loss) used in SimCLR.
- `main.py`: The main script to run the training process.
- `models.py`: Defines the SimCLR model architecture, including the encoder and projection head.
- `README.md`: Provides an overview of the project, including installation, usage, and details about the implementation.
- `requirements.txt`: Lists the dependencies and packages required to run the project.
- `training.py`: Contains the training loop for the SimCLR model.
- `utils.py`: Utility functions used throughout the project.
- `visualization.py`: Provides functions for visualizing embeddings using t-SNE.
