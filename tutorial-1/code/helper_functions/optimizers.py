import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Callable
from .Layers import Linear

class optimizer_info():
    @property
    def learning_rate(self):
        return self.lr

    @property
    def optimizer_type(self):
        return self._optimizer_type

    @property
    def optimizer_name(self):
        return self._name
    
    @property
    def specification(self):
        return self._specification

    def __str__(self):
        s = f"These are the specifications of the optimizer : '{self._optimizer_type}' \n"
        s += "\n".join([f"{str(x)} : {str(y)}" for x,y in self._specification.items()])
        return s

class SGD(optimizer_info):
    def __init__(self, lr : float,
                       weight_decay : float,
                       name = "default_name", **kwargs):
        self._name = name
        self._optimizer_type = "SGD"

        self.lr = lr
        self.weight_decay = weight_decay

        self._specification = {"lr" : lr, "weight_decay": weight_decay} # This will tell what are the specifications of different optimizers
    
    def update_params(self, obj : Linear):

        # getting the necessary variables
        dw = obj.optimizer_variables['dw'] + (self.weight_decay)*obj.w

        # updating the parameters here
        obj.w -= self.lr*dw

class GD_Momentum(optimizer_info):
    def __init__(self, lr : float, 
                       beta : float,
                       weight_decay : float,
                       name = "default_name", **kwargs):
        
        self._name = name
        self._optimizer_type = "GD_with_Momentum"

        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta

        self._specification = {"lr" : self.lr, "beta" : self.beta, "weight_decay" : weight_decay,
                               "du" : "previous gradients updates for momentum"} # This will tell what are the specifications of different optimizers
    
    def _init_params_(self, obj : Linear):
        # To initialize all the necessary things needed for momentum in the objects optimizer_variables

        # getting the necessary variables
        dw = obj.optimizer_variables['dw']
        if "du" not in obj.optimizer_variables: 
            obj.optimizer_variables['du'] = np.zeros(shape = dw.shape, dtype = np.float64)
    
    def update_params(self, obj : Linear):
        
        # initializing the all the params
        self._init_params_(obj)

        # getting the necessary variables
        dw = obj.optimizer_variables['dw'] + (self.weight_decay)*obj.w
        ut = (self.beta)*obj.optimizer_variables['du'] + dw

        # updating the parameters here and optimizer variables
        obj.w -= self.lr*ut
        obj.optimizer_variables['du'] = ut

class NAG(optimizer_info):
    def __init__(self, lr : float, 
                       beta : float,
                       weight_decay : float,
                       name = "default_name", **kwargs):
        
        self._name = name
        self._optimizer_type = "NAG"

        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta

        self._specification = {"lr" : self.lr, "beta" : self.beta, "weight_decay" : weight_decay, "du" : "previous gradients updates for momentum"} # This will tell what are the specifications of different optimizers
    
    def _init_params_(self, obj : Linear):
        # To initialize all the necessary things needed for momentum in the objects optimizer_variables

        # getting the necessary variables
        dw = obj.optimizer_variables['dw']
        if "du" not in obj.optimizer_variables: 
            obj.optimizer_variables['du'] = np.zeros(shape = dw.shape, dtype = np.float64)
    
    def update_params(self, obj : Linear):
        
        # initializing the all the params
        self._init_params_(obj)

        # getting the necessary variables
        dw = obj.optimizer_variables['dw'] + (self.weight_decay)*obj.w
        ut = (self.beta)*obj.optimizer_variables['du'] + dw

        # updating the parameters here and optimizer variables
        obj.w += -(self.lr*dw) - (self.lr*self.beta*ut)
        obj.optimizer_variables['du'] = ut

class RMSprop(optimizer_info):
    def __init__(self, lr : float, 
                       beta : float,
                       epsilon : float,
                       weight_decay : float,
                       name = "default_name", **kwargs):
        
        self._name = name
        self._optimizer_type = "RMSprop"

        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.epsilon = epsilon

        self._specification = {"lr" : self.lr, 
                               "beta" : self.beta, 
                               "epsilon" : self.epsilon,
                               "weight_decay" : weight_decay,
                               "dv" : "cumulative square of gradients"} # This will tell what are the specifications of different optimizers

    def _init_params_(self, obj : Linear):
        # To initialize all the necessary things needed for momentum in the objects optimizer_variables

        # getting the necessary variables
        dw = obj.optimizer_variables['dw']
        if "dv" not in obj.optimizer_variables: 
            obj.optimizer_variables['dv'] = np.zeros(shape = dw.shape, dtype = np.float64)
    
    def update_params(self, obj : Linear):
        
        # initializing the all the params
        self._init_params_(obj)

        # getting the necessary variables
        dw = obj.optimizer_variables['dw'] + (self.weight_decay)*obj.w
        vt = (self.beta)*obj.optimizer_variables['dv'] + (1 - self.beta)*(dw**2)

        # updating the parameters here and optimizer variables
        obj.w -= (self.lr)*(dw)/((vt + self.epsilon)**0.5)
        obj.optimizer_variables['dv'] = vt

class Adam(optimizer_info):
    def __init__(self, lr : float, 
                       beta1 : float,
                       beta2 : float,
                       epsilon : float,
                       weight_decay : float,
                       name = "default_name", **kwargs):
        
        self._name = name
        self._optimizer_type = "Adam"

        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self._specification = {"lr" : self.lr, 
                               "beta1" : self.beta1, 
                               "beta2" : self.beta2,
                               "epsilon" : self.epsilon,
                               "weight_decay" : weight_decay,
                               "dm" : "classical momentum",
                               "dv" : "cumulative square of gradients"} # This will tell what are the specifications of different optimizers
    
    def _init_params_(self, obj : Linear):
        # To initialize all the necessary things needed for momentum in the objects optimizer_variables

        # getting the necessary variables
        dw = obj.optimizer_variables['dw']

        if "dv" not in obj.optimizer_variables: 
            obj.optimizer_variables['dv'] = np.zeros(shape = dw.shape, dtype = np.float64)

        if "dm" not in obj.optimizer_variables: 
            obj.optimizer_variables['dm'] = np.zeros(shape = dw.shape, dtype = np.float64)
        
        if "beta1^t" not in obj.optimizer_variables:
            obj.optimizer_variables['beta1^t'] = self.beta1

        if "beta2^t" not in obj.optimizer_variables:
            obj.optimizer_variables['beta2^t'] = self.beta2
    
    def update_params(self, obj : Linear):
        
        # initializing the all the params
        self._init_params_(obj)

        # getting the necessary variables
        dw = obj.optimizer_variables['dw'] + (self.weight_decay)*obj.w
        mt = (self.beta1)*obj.optimizer_variables['dm'] +  (1 - self.beta1)*(dw)
        vt = (self.beta2)*obj.optimizer_variables['dv'] + (1 - self.beta2)*(dw**2)
        beta1_t = obj.optimizer_variables['beta1^t']
        beta2_t = obj.optimizer_variables['beta2^t']

        mt_normalized = mt/(1 - beta1_t)
        vt_normalized = vt/(1 - beta2_t)

        # updating the parameters here and optimizer variables
        obj.w -= (self.lr)*(mt_normalized)/((vt_normalized**0.5) + self.epsilon)

        obj.optimizer_variables['dv'] = vt
        obj.optimizer_variables['dm'] = mt
        obj.optimizer_variables['beta1^t'] = self.beta1*beta1_t
        obj.optimizer_variables['beta2^t'] = self.beta2*beta2_t

class Nadam(optimizer_info):
    def __init__(self, lr : float, 
                       beta1 : float,
                       beta2 : float,
                       epsilon : float,
                       weight_decay : float,
                       name = "default_name", **kwargs):
        
        self._name = name
        self._optimizer_type = "Nadam"

        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self._specification = {"lr" : self.lr, 
                               "beta1" : self.beta1, 
                               "beta2" : self.beta2,
                               "epsilon" : self.epsilon,
                               "weight_decay" : weight_decay,
                               "dm" : "classical momentum",
                               "dv" : "cumulative square of gradients"} # This will tell what are the specifications of different optimizers
    
    def _init_params_(self, obj : Linear):
        # To initialize all the necessary things needed for momentum in the objects optimizer_variables

        # getting the necessary variables
        dw = obj.optimizer_variables['dw']

        if "dv" not in obj.optimizer_variables: 
            obj.optimizer_variables['dv'] = np.zeros(shape = dw.shape, dtype = np.float64)

        if "dm" not in obj.optimizer_variables: 
            obj.optimizer_variables['dm'] = np.zeros(shape = dw.shape, dtype = np.float64)
        
        if "beta1^t" not in obj.optimizer_variables:
            obj.optimizer_variables['beta1^t'] = self.beta1

        if "beta2^t" not in obj.optimizer_variables:
            obj.optimizer_variables['beta2^t'] = self.beta2
    
    def update_params(self, obj : Linear):
        
        # initializing the all the params
        self._init_params_(obj)

        # getting the necessary variables
        dw = obj.optimizer_variables['dw'] + (self.weight_decay)*obj.w
        mt = (self.beta1)*obj.optimizer_variables['dm'] +  (1 - self.beta1)*(dw)
        vt = (self.beta2)*obj.optimizer_variables['dv'] + (1 - self.beta2)*(dw**2)
        beta1_t = obj.optimizer_variables['beta1^t']
        beta2_t = obj.optimizer_variables['beta2^t']

        mt_normalized = mt/(1 - beta1_t)
        vt_normalized = vt/(1 - beta2_t)

        # updating the parameters here and optimizer variables
        obj.w -= (self.lr)*((self.beta1*mt_normalized) + (1-self.beta1)*dw/(1 - beta1_t) )/((vt_normalized**0.5) + self.epsilon)

        obj.optimizer_variables['dv'] = vt
        obj.optimizer_variables['dm'] = mt
        obj.optimizer_variables['beta1^t'] = self.beta1*beta1_t
        obj.optimizer_variables['beta2^t'] = self.beta2*beta2_t