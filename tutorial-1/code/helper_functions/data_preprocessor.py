import numpy as np
import numpy.typing as npt

def one_hot_encode(Y, num_classes=10):
    
    num_samples = Y.shape[0]
    Y_one_hot = np.zeros((num_samples, num_classes), dtype=np.float64)
    Y_one_hot[np.arange(num_samples), Y] = 1  
    return Y_one_hot.astype(np.float64)

def train_val_split(X, Y, train_ratio=0.9, seed=None, shuffle = True):
    X = X.astype(np.float64)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility
    if shuffle : np.random.shuffle(indices)  # Shuffle dataset indices

    train_size = int(num_samples * train_ratio)
    val_size = num_samples - train_size
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    return X[train_idx].reshape(train_size,-1), one_hot_encode(Y[train_idx]), X[val_idx].reshape(val_size,-1), one_hot_encode(Y[val_idx])