import numpy as np
import numpy.typing as npt
from .model_builder import NN
from .Layers import *
from .Losses import *
from .optimizers import *
from .training_loop_builder import Training_Loop

class automated_trainer():
    def __init__(self,
                 train_x, train_y, val_x, val_y, 
                 input_size : int = 28*28,
                 output_size : int = 10,
                 epochs : int = 5,
                 n_hidden : int = 3,
                 hidden_size : int = 32,
                 weight_decay : float = 0,
                 lr : float = 1e-3,
                 optimizer : str = "adam",
                 batch_size : int = 16,
                 weight_initialization : str = "xavier",
                 activation_fn : str = "relu",
                 beta : float = 0.9,
                 beta1 : float = 0.9,
                 beta2 : float = 0.999,
                 epsilon : float = 1e-7,
                 loss : str = "cross_entropy" ,
                 **kwargs,
                ):
        
        self.epochs = epochs
        self.batch_size = batch_size
        # self.n_hidden = n_hidden

        # getting the init fn
        self.initializer_map = {'xavier' : xavier_init, 'random' : random_init_uniform}
        self.init_fn = self.initializer_map[weight_initialization.lower()]

        # getting the activation fn
        self.activation_fn_map = {"sigmoid" : Sigmoid, "tanh" : Tanh, "relu" : ReLU}
        self.act_fn = self.activation_fn_map[activation_fn.lower()]

        # for optimizer and loading the optimizer
        self.optimizer_map = {"sgd" : SGD, "momentum" : GD_Momentum, "nesterov" : NAG, 
                              "rmsprop" : RMSprop, "adam" : Adam, "nadam" : Nadam}
        self.optimizer_params = {"lr" : lr, "weight_decay" : weight_decay, "beta" : beta, 
                                 "beta1" : beta1, "beta2" : beta2, "epsilon" : epsilon}
        
        self.optimizer_class = self.optimizer_map[optimizer.lower()]
        print(f"optimizer loaded with this setting : \n{self.optimizer_class(**self.optimizer_params)} \n")

        # loading the loss function
        self.loss_map = {"cross_entropy" : softmax_cross_entropy, "mean_squared_error" : L2_Loss}
        self.loss_class = self.loss_map[loss.lower()]

        # creating nn
        self.__make_nn__(input_size = input_size, output_size = output_size, n_hidden = n_hidden, hidden_size = hidden_size)

        # creating the trainer
        self.trainer = Training_Loop(train_x = train_x,
                                    train_y = train_y,
                                    val_x = val_x,
                                    val_y = val_y,
                                    Network = self.nn)

    def __make_nn__(self, input_size, n_hidden, hidden_size, output_size,):

        arch = [ Linear(input_dim = input_size, output_dim = hidden_size, initializer_function = self.init_fn) ]

        # creating the network
        for i in range(n_hidden):
            arch.append(self.act_fn())
            arch.append(Linear(input_dim = hidden_size, output_dim = hidden_size, initializer_function = self.init_fn))
        
        arch.append(self.act_fn())
        arch.append(Linear(input_dim = hidden_size, output_dim = output_size, initializer_function = self.init_fn))
    

        self.nn = NN(arch = arch, optimizer = self.optimizer_class, 
                     optimizer_params = self.optimizer_params, Loss = self.loss_class())
        
        print(f"NN created successfully !, total params : {self.nn.parameters_count*1e-3:.3f}K \n")
    
    def train_network(self):
        stats = self.trainer.train(epochs = self.epochs,
                           batch_size = self.batch_size,)
        return stats