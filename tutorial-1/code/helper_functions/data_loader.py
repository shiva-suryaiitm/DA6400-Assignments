from keras.datasets import fashion_mnist
from keras.datasets import mnist
from .data_preprocessor import train_val_split

def fashion_mnist_loader(seed = None,  shuffle = True):

    (fashion_mnist_x_train, fashion_mnist_y_train), (fashion_mnist_x_test, fashion_mnist_y_test) = fashion_mnist.load_data()
    train_x, train_y, val_x, val_y = train_val_split(X = fashion_mnist_x_train, Y = fashion_mnist_y_train, 
                                                     train_ratio=0.9, seed=seed, shuffle=shuffle)
    
    return (train_x, train_y, val_x, val_y, fashion_mnist_x_test, fashion_mnist_y_test)

def mnist_loader(seed = None, shuffle = True):
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
    train_x, train_y, val_x, val_y = train_val_split(X = mnist_x_train, Y = mnist_y_train, 
                                                     train_ratio=0.9, seed=seed, shuffle=shuffle)
    
    return (train_x, train_y, val_x, val_y, mnist_x_test, mnist_y_test)