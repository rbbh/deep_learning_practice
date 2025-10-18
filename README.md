# CIFAR-10 Classification with CNN and Vision Transformer

This project provides a framework for training and evaluating two deep learning models, a VGG-style Convolutional Neural Network (CNN) and a Vision Transformer (ViT), on the CIFAR-10 dataset. The project is structured to be easily extensible and configurable.

## Project Structure

```
.
├── config
│   └── default_conf.yaml
├── data
│   └── data_loader
│       ├── __init__.py
│       └── data_loader.py
├── main.py
├── models
│   ├── __init__.py
│   ├── cnn_model.py
│   └── vit_model.py
├── notebooks
│   └── demonstration.ipynb
├── README.md
├── requirements.txt
├── testing
│   ├── __init__.py
│   └── tester.py
├── training
│   ├── __init__.py
│   └── trainer.py
└── utils
    ├── __init__.py
    └── utils.py
```

### Directory Description

- **`config/`**: Contains configuration files in YAML format.
- **`data_loader/`**: Handles data loading and preprocessing. The CIFAR-10 dataset will be downloaded here.
- **`models/`**: Contains the PyTorch implementations of the deep learning models.
- **`notebooks/`**: Jupyter notebooks for demonstration and analysis.
- **`testing/`**: Contains the code for evaluating the trained models.
- **`training/`**: Contains the model training logic.
- **`utils/`**: Contains utility functions, such as file readers.
- **`main.py`**: The main script to execute the training and testing pipeline.
- **`requirements.txt`**: A list of all the Python packages required to run the project.

## Getting Started

### Prerequisites

- Python 3.8+
- Pip

### Installation

1.  Clone the repository:
    ```bash
    git clone git@github.com:rbbh/deep_learning_practice.git
    cd deep_learning_practice
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training and Testing

To train and test the models, run the `main.py` script:

```bash
python main.py
```

You can also specify a different configuration file using the `--config` argument:

```bash
python main.py --config path/to/your/config.yaml
```

### Jupyter Notebook

A Jupyter notebook is provided for an interactive demonstration. To use it, make sure you have Jupyter installed (`pip install jupyter`) and run:

```bash
jupyter notebook notebooks/practice_cnn.ipynb
```

## Configuration

The `config/default_conf.yaml` file contains the default parameters for the data, models, training, and testing. You can modify this file to change hyperparameters like learning rate, batch size, number of epochs, etc.

```yaml
# Data configuration
data:
  image_size: 32
  batch_size: 64

# CNN Model configuration
cnn_model:
  epochs: 5
  lr: 0.001

# ViT Model configuration
vit_model:
  epochs: 5
  lr: 0.001
  embed_dim: 128
  depth: 4
  n_heads: 4
  mlp_ratio: 4.0
  qkv_bias: True
  p: 0.0
  attn_p: 0.0
```
