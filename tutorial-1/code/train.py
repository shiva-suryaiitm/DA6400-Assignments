import argparse
import wandb
from helper_functions import automated_trainer
from helper_functions import fashion_mnist_loader, mnist_loader

def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network with WandB integration")
    
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str,  default="myname", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset choice")
    parser.add_argument("-e", "--epochs", type=int,     default=10, help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size used to train neural network")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="nadam", help="Optimizer choice")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float,       default=0.9, help="Momentum used by momentum and nag optimizers")
    parser.add_argument("--beta", type=float,   default=0.9, help="Beta used by rmsprop optimizer")
    parser.add_argument("--beta1", type=float,  default=0.9, help="Beta1 used by adam and nadam optimizers")
    parser.add_argument("--beta2", type=float,  default=0.999, help="Beta2 used by adam and nadam optimizers")
    parser.add_argument("--eps", "--epsilon", type=float,     default=1e-7, help="Epsilon used by optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay used by optimizers")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int,     default=5, help="Number of hidden layers in the feedforward neural network")
    parser.add_argument("-sz", "--hidden_size", type=int,     default=128, help="Number of hidden neurons in a feedforward layer")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="ReLU", help="Activation function")
    
    return parser.parse_args()

def train_model(args, trainer_wandb : automated_trainer):

        # Initialize wandb
        wandb.login(key = args['wandb_entity'])
        run = wandb.init(project = args['wandb_project'], config = args)
        config = wandb.config
        run.name = f"act_fn : {config.activation_fn} _ hidden : {config.n_hidden}-{config.hidden_size} _ optimizer : {config.optimizer} "

        # training the network
        stats = trainer_wandb.train_network()

        # logging the inputs
        for i in range(0,len(stats['Train_loss'])):

            wandb.log({
                    "epoch": i+1,
                    "train_accuracy" : stats['Train_accuracy'][i],
                    "train_loss": stats['Train_loss'][i],
                    "val_loss" : stats['Validation_loss'][i],
                    "val_accuracy": stats['Validation_accuracy'][i],
                })

        wandb.finish()

if __name__ == "__main__":

    args = get_args()
    trainer_args = vars(args)
    trainer_args['n_hidden'] = args.num_layers
    trainer_args['weight_initialization'] = args.weight_init
    trainer_args['activation_fn'] = args.activation
    trainer_args['lr'] = args.learning_rate 

    if args.dataset == "mnist":
        train_x, train_y, val_x, val_y, test_x, test_y = mnist_loader()
    elif args.dataset == "fashion_mnist":
        train_x, train_y, val_x, val_y, test_x, test_y = fashion_mnist_loader()

    """ args required for automated_trainer : 
    train_x, train_y, val_x, val_y, 
                 input_size : int = 28*28, yes
                 output_size : int = 10, yes
                 epochs : int = 5, yes
                 n_hidden : int = 3, yes/added
                 hidden_size : int = 32, yes
                 weight_decay : float = 0, yes
                 lr : float = 1e-3, yes
                 optimizer : str = "adam", yes
                 batch_size : int = 16, yes
                 weight_initialization : str = "xavier", yes/added
                 activation_fn : str = "relu", yes/added
                 beta : float = 0.9, yes
                 beta1 : float = 0.9, yes
                 beta2 : float = 0.999, yes
                 epsilon : float = 1e-7, yes
                 loss : str = "cross_entropy", yes

    """

    print(args)

    ## creating automated loader
    NN_trainer = automated_trainer(train_x = train_x,
                                   train_y = train_y,
                                   val_x = val_x,
                                   val_y = val_y,
                                   **trainer_args,)

    train_model(trainer_args, NN_trainer)    