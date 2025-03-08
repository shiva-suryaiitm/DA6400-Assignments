import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Callable


class Layer_Info():
    @property
    def is_activation(self): 
        return self._is_activation

    @property
    def layer_name(self):
        return self._name

    @property
    def layer_type(self):
        return self._layer_type

class softmax_cross_entropy(Layer_Info):
    def __init__(self, name = "default_name"):
        self._name = name
        self._is_activation = False
        self._layer_type = "Softmax-with-Cross-Entropy"
    
    def forward(self, x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        # for numerical stability
        self.x = x
        x_max = np.max(x, axis=1, keepdims=True)  # size -> (batch_size, 1)
        exp_x = np.exp(x - x_max)  # Subtract max for numerical stability
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True) # normalizing along the axis
        # raise Exception(f"Test 5 : Inside the Softmax Cross entropy , output shape : {self.output.shape}") # -> passed
        return self.output

    def calculate_loss(self, y_true : npt.NDArray[np.float64]) -> float:
        self.loss = np.sum(-np.log(self.output + 1e-10)*y_true, axis = 1)
        return np.mean(self.loss)

    def backward(self, y_true : npt.NDArray[np.float64]) -> npt.NDArray[np.float64] :
        self.dx = -(y_true - self.output)
        return self.dx

class L2_Loss(Layer_Info):
    def __init__(self, name = "default_name"):
        self._name = name
        self._is_activation = False
        self._layer_type = "L2-Loss"
    
    def forward(self, x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.x = x
        self.output = self.x.copy()
        return self.output

    def calculate_loss(self, y_true : npt.NDArray[np.float64]) -> float:
        self.loss = np.mean((y_true - self.output)**2, axis = 1)
        return np.mean(self.loss)

    def backward(self, y_true : npt.NDArray[np.float64])  -> npt.NDArray[np.float64] :
        self.dx = -(y_true - self.output)
        return self.dx

class Sigmoid_with_BCE(Layer_Info):
    def __init__(self, name = "default_name"):
        self._name = name
        self._is_activation = False
        self._layer_type = "Softmax-with-Cross-Entropy"
    
    def forward(self, x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Question didn't ask for it, so haven't implemented, just defined for namesake")

    def backward(self, y_true : npt.NDArray[np.float64])  -> npt.NDArray[np.float64] :
        raise NotImplementedError("Question didn't ask for it, so haven't implemented, just defined for namesake")