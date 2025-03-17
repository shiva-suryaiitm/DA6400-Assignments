# DA6401 Assignment 1 Documentation

This document provides an overview of the logic, code structure, and functionalities of each Python file in the neural network project.

[Github repo link](https://github.com/shiva-suryaiitm/DA6400-Assignments)

[Wandb report link](https://wandb.ai/shivasurya-iit-madras/DA6401-fashion-mnist-assig-1/reports/DA6401-Assignment-1-PH21B009-SHIVASURYA--VmlldzoxMTcwMTYxMg?accessToken=2b64stniltt9m0cvxm13m02m0odilgfik9rj7usn9a1bh1wd7roub3sp5w462aym)

---

## 📁 Project Structure Overview

| File Name                      | Description                                              |
|--------------------------------|----------------------------------------------------------|
| `data_preprocessor.py`         | Data preprocessing utilities (one-hot encoding, splitting)|
| `data_loader.py`               | Functions for loading MNIST and Fashion-MNIST datasets   |
| `Layers.py`                    | Neural network layers and activation functions           |
| `Losses.py`                    | Loss functions (Cross Entropy, Mean Squared Error)       |
| `optimizers.py`                | Optimization algorithms (SGD, Adam, RMSprop, etc.)       |
| `model_builder.py`             | Neural network model definition and management           |
| `training_loop_builder.py`     | Training and inference loop implementations              |
| `automated_trainer_builder.py` | Automated trainer for simplified model setup and training|

---

## 📝 Detailed File Documentation

### 1. Data Preprocessing (`data_preprocessor.py`)
- **Functions:**
  - `one_hot_encode(Y, num_classes=10)`: Converts labels to one-hot vectors.
  - `train_val_split(X, Y, train_ratio=0.9, seed=None, shuffle=True)`: Splits dataset into training and validation sets.

---

### 2. Data Loading (`data_loader.py`)
- **Functions:**
  - `fashion_mnist_loader(seed=None, shuffle=True)`: Loads Fashion-MNIST dataset.
  - `mnist_loader(seed=None, shuffle=True)`: Loads MNIST dataset.

---

### 3. Layers (`Layers.py`)
- **Initialization Functions:**
  - `xavier_init(n_in, n_out)`
  - `random_init_uniform(n_in, n_out)`

- **Layer Classes:**
  - `Linear`: Fully connected layer with forward/backward propagation. Also contains step method in which the given optimizer will update the parameters of the Layer
  - Activation Layers: `Sigmoid`, `Tanh`, `ReLU`. (Contains forward and backward methods) 

---

### 4. Loss Functions (`Losses.py`)
- **Classes:**
  - `softmax_cross_entropy`: Softmax activation combined with cross-entropy loss; Contains forward (softmax calc), backward and loss calculation function
  - `L2_Loss`: Mean Squared Error loss function. 

---

### 5. Optimizers (`optimizers.py`)
- **Implemented Optimizers:**
  - Stochastic Gradient Descent (`SGD`)
  - Momentum-based Gradient Descent (`GD_Momentum`)
  - Nesterov Accelerated Gradient (`NAG`)
  - RMSprop
  - Adam
  - Nadam

Each optimizer includes parameter initialization and update methods. Optimizer stores the necessary information need for updating the weights of the Linear class, eg : Momentum will store previous calculated momentum inside the self.optimizer_variables dict in Linear class

---

### 6. Model Builder (`model_builder.py`)
- **Class:** 
  - `NN`: Manages neural network architecture, forward/backward propagation, loss calculation, parameter updates.
---

### 7. Training & Inference Loops (`training_loop_builder.py`)
- **Classes:**
  - `Training_Loop`: Handles training iterations over epochs and batches.
    - After loading the optimizer prints them for debugging purposes
    - Tracks training/validation loss and accuracy.
    - Uses batch generators for efficient data handling.
  
  - `Inference_Loop`: Generates predictions on test data in batches.

---

### 8. Automated Trainer (`automated_trainer_builder.py`)
- **Class:** 
  - `automated_trainer`: Simplifies neural network creation and training with customizable parameters:
    - train and val data
    - Number of hidden layers
    - Hidden layer size
    - Activation function
    - Weight initialization method
    - Optimizer type and parameters
    - Loss function

---

## Quick Start Example

## Usage

```python
from data_loader import fashion_mnist_loader
from automated_trainer_builder import automated_trainer

# Load dataset
train_x, train_y, val_x, val_y, test_x, test_y = fashion_mnist_loader(seed=42)

# Initialize automated trainer with custom settings
trainer = automated_trainer(
    train_x=train_x,
    train_y=train_y,
    val_x=val_x,
    val_y=val_y,
    epochs=10,
    n_hidden=2,
    hidden_size=64,
    lr=0.001,
    optimizer='adam',
    activation_fn='relu',
    weight_initialization='xavier',
    loss='cross_entropy'
)

# Train the model
training_stats = trainer.train_network()
```

---

## Workflow Summary

1. Load data using functions in `data_loader.py`.
2. Preprocess data using utilities in `data_preprocessor.py`.
3. Define neural network architecture using classes from `Layers.py`.
4. Select a loss function from available options in `Losses.py`.
5. Choose an optimizer from provided implementations in `optimizers.py`.
6. Build your model using the class defined in `model_builder.py`.
7. Train your model using loops defined in `training_loop_builder.py`.
8. Alternatively, use the automated trainer (`automated_trainer_builder.py`) for simplified setup.

---

### 9. Training Script (`train.py`)

The `train.py` script serves as the entry point for training neural networks with configurable parameters via command-line arguments and integrates with Weights & Biases (`wandb`) for experiment tracking and visualization.

**Main functionalities:**

- **Command-line Argument Parsing**:
  - Uses Python's `argparse` library to handle various training parameters:
    - WandB project and entity names for experiment tracking.
    - Dataset selection (`mnist` or `fashion_mnist`).
    - Number of epochs, batch size, learning rate, optimizer choice, loss function, activation function, weight initialization method, number of hidden layers, hidden layer size, and other hyperparameters.

- **Integration with WandB (Weights & Biases)**:
  - Initializes WandB logging to track training metrics such as accuracy and loss.
  - Logs metrics at each epoch to the WandB dashboard.

- **Data Loading**:
  - Loads the selected dataset (`mnist` or `fashion_mnist`) using functions from `data_loader.py`.

- **Model Training**:
  - Utilizes the `automated_trainer` class from `automated_trainer_builder.py` to build and train the neural network based on provided arguments.

- **Logging Metrics**:
  - Logs training and validation accuracy and loss metrics for each epoch to WandB.

---

## Usage Example

Run training from the command line:
```
python train.py
--wandb_project "myprojectname"
--wandb_entity "myname"
--dataset "fashion_mnist"
--epochs 10
--batch_size 64
--loss "cross_entropy"
--optimizer "nadam"
--learning_rate 0.001
--num_layers 5
--hidden_size 128
--activation "ReLU"
--weight_init "Xavier"
```


This command trains a neural network on the Fashion-MNIST dataset for 10 epochs using Nadam optimizer, ReLU activation, cross-entropy loss, and logs training metrics to WandB.

---
