import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Callable
from .Layers import Linear, Sigmoid, Tanh, ReLU
from .Losses import softmax_cross_entropy, L2_Loss

class NN():
    def __init__(self, arch : List[Linear | Sigmoid | Tanh | ReLU],
                       optimizer,
                       optimizer_params : dict,
                       Loss : softmax_cross_entropy | L2_Loss | None,
                       arch_name : str = "default_name"):
        
        # Architecture is a series of layers 
        self.arch = arch
        self.arch_name = arch_name
        self.optimizer_params = optimizer_params
        self.optimizer = optimizer(**self.optimizer_params)
        self.Loss = Loss

        # other variables for the network
        self.total_parameters = -1

    @property
    def parameters_count(self):
        self.total_parameters = 0
        
        for layer in self.arch:
            if layer.is_activation == False: 
                self.total_parameters += (layer.w.shape[0] * layer.w.shape[1])
        
        return self.total_parameters

    def forward(self, input : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # passing through all layers in the architecture
        curr_output = input
        
        for i,layer in enumerate(self.arch):
            curr_output = layer.forward(curr_output)

            # updating names, if names are not given
            if layer.layer_name == "default_name": layer._name = f"{layer.layer_type} : {i+1}"

        # finally passing through the output for the Loss (this is not the loss, just the activation fn for the loss)
        output = self.Loss.forward(curr_output) if self.Loss != None else curr_output

        return output
    
    def calculate_loss(self, y_true : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.Loss.calculate_loss(y_true)

    def backward(self, y_true : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        # getting the gradient from the Loss
        prev_layer_grad = self.Loss.backward(y_true)

        # backward pass through all layers (iterating in opposite direction)
        for layer in self.arch[::-1]:
            # print(prev_layer_grad.shape)
            prev_layer_grad = layer.backward(prev_layer_grad)
    
    def step(self,):
        # updating the layer parameters if it is not activation layer
        for layer in self.arch[::-1]:
            if layer.is_activation == False: 
                layer.step(self.optimizer)