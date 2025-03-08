import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Callable

# initialization fn
def xavier_init(n_in, n_out):
    """ Xavier (Glorot) Initialization """
    limit = np.sqrt(6 / (n_in + n_out))  # Computing the limit
    return np.random.uniform(-limit, limit, size=(n_in, n_out)).astype(np.float64)

def random_init_uniform(n_in, n_out):
    """ Random Initialization (Uniform) """
    return np.random.uniform(-1, 1, size=(n_in, n_out)).astype(np.float64)


"""Layer functions"""

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

def matrix_mul(x,y):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y) 
    return x@y

class Linear(Layer_Info):
    def __init__(self, input_dim : int, 
                       output_dim : int, 
                       initializer_function : Callable = xavier_init,
                       bias : bool = True,
                       name = "default_name"):

        # getting the weights of the network (adding one more input for bias)
        self.input_dim = input_dim + int(bias)
        self.output_dim = output_dim
        self.w = initializer_function(self.input_dim, self.output_dim)

        # other necessary variables for the info the layer
        self.is_bias = bias
        self._name = name
        self._layer_type = "Linear"
        self._is_activation = False

        # variables for gradients and used by optimizers
        self.dw = np.zeros(shape = self.w.shape, dtype = np.float64)
        self.optimizer_variables = {} # -> This is important thing used used by the optimizers for updating the weights
    
    def forward(self, x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        # changing the dtype and adding the input for the bias
        self.x = x.astype(np.float64)
        if self.is_bias: self.x = np.concatenate((self.x,np.ones(shape = (x.shape[0],1), dtype = np.float64)), axis = 1)

        # returning the matrix multiplication
        # raise Exception(f"""Test 1 : print under forward of linear , input_shape : {self.x.shape} , weight_shape : {self.w.shape}
        #                 output_shape : {(self.x@self.w).shape}""") # -> Test passed
    
        # return self.x@self.w
        return matrix_mul(self.x,self.w)
        
    def backward(self, prev_grad : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        # updating the gradient for weights and x
        # self.dw = self.x.T@prev_grad
        # self.dx = prev_grad@self.w.T
        self.dw = matrix_mul(self.x.T,prev_grad)
        self.dx = matrix_mul(prev_grad,self.w.T)
        if self.is_bias: self.dx = self.dx[:,:-1] #( We have to remove the extra bias term we have added for calculation)

        # this will be used by the optimizer
        self.optimizer_variables['dw'] = self.dw
        self.optimizer_variables['dx'] = self.dx

        return self.dx
    
    def step(self, optimizer):
        # for updating the values
        optimizer.update_params(self)

class Sigmoid(Layer_Info):
    def __init__(self, name = "default_name"):
        self._name = name
        self._layer_type = "Sigmoid"
        self._is_activation = True
    
    def forward(self, x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        # for numerical stability
        self.x = x
        x = np.clip(x, -100, 100)
        self.output = np.where( x >= 0, 1 / (1 + np.exp(-x)),  np.exp(x) / (1 + np.exp(x)) )
        # raise Exception(f"Test 2 : Inside the sigmoid act fn , output shape : {self.output.shape}") 
        return self.output

    def backward(self, prev_grad : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        # updating the gradient for x (using hadamard product)
        self.dx = prev_grad*self.output*(1 - self.output)
        return self.dx
    

class Tanh(Layer_Info):
    def __init__(self, name = "default_name"):
        self._name = name
        self._layer_type = "Tanh"
        self._is_activation = True
    
    def forward(self, x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        # for numerical stability
        self.x = x
        self.output = np.tanh(self.x)
        # self.output = np.where( x >= 0, (1 - np.exp(-2*x))/(np.exp(-2*x) + 1),  (np.exp(2*x) - 1)/(1 + np.exp(2*x)) )
        # raise Exception(f"Test 3 : Inside the Tanh act fn , output shape : {self.output.shape}")
        return self.output

    def backward(self, prev_grad : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # updating the gradient for x (using hadamard product)
        self.dx = prev_grad*(1 - self.output**2)
        return self.dx

class ReLU(Layer_Info):
    def __init__(self, name = "default_name"):
        self._name = name
        self._layer_type = "ReLU"
        self._is_activation = True
    
    def forward(self, x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        # for numerical stability
        self.x = x
        self.output = np.maximum(x,0)
        # raise Exception(f"Test 4 : Inside the ReLU act fn , output shape : {self.output.shape}") # -> passed
        return self.output

    def backward(self, prev_grad : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # updating the gradient for x (using hadamard product)
        self.dx = prev_grad*(self.output > 0)
        return self.dx