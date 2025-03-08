import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from .model_builder import NN

class Training_Loop():
    def __init__(self, train_x : npt.NDArray, train_y : npt.NDArray,
                       val_x : npt.NDArray, val_y : npt.NDArray, 
                       Network : NN,):
        
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.network = Network
    
    def batch_generator(self, X, Y, batch_size, shuffle=True):
        num_samples = X.shape[0]
        indices = np.arange(num_samples)

        if shuffle:
            np.random.shuffle(indices)  # Shuffle indices for randomness

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], Y[batch_indices]

    def train(self, epochs : int, batch_size : int) -> dict:

        total_train_loss = []
        total_train_accuracy = []

        total_validation_loss = []
        total_validation_accuracy = []  

        desc = ""
        loop_obj = range(epochs)
        for e in loop_obj:

            epoch_loss = 0
            epoch_correct = 0

            # Train loop of the model
            for X_batch, Y_true in tqdm(self.batch_generator(self.train_x, self.train_y, batch_size), desc = f"train-progress-bar , epcoh : {e+1}"):

                # getting the output
                Y_pred = self.network.forward(X_batch)
                epoch_loss += self.network.calculate_loss(Y_true)

                # updating the parameters
                self.network.backward(y_true = Y_true)
                self.network.step()

                # getting the total number of correct answers
                Y_pred_class = np.argmax(Y_pred, axis = 1)
                Y_true_class = np.argmax(Y_true, axis = 1)
                epoch_correct += np.sum(Y_pred_class == Y_true_class)
            
            total_train_loss.append(epoch_loss)
            total_train_accuracy.append(epoch_correct/self.train_x.shape[0])

            # Validation loop of the model
            epoch_loss_val = 0
            epoch_correct_val = 0
            for X_batch, Y_true in self.batch_generator(self.val_x, self.val_y, batch_size):

                # getting the output
                Y_pred = self.network.forward(X_batch)
                epoch_loss_val += self.network.calculate_loss(Y_true)

                # getting the total number of correct answers
                Y_pred_class = np.argmax(Y_pred, axis = 1)
                Y_true_class = np.argmax(Y_true, axis = 1)
                epoch_correct_val += np.sum(Y_pred_class == Y_true_class)
            
            total_validation_loss.append(epoch_loss_val)
            total_validation_accuracy.append(epoch_correct_val/self.val_x.shape[0])

            # updating desc for tqdm
            desc = f"Train error : {epoch_loss:.4f} ; Train acc : {(epoch_correct/self.train_x.shape[0]):.4f} ; val error : {epoch_loss_val:.4f}  ; val acc : {(epoch_correct_val/self.val_x.shape[0]):.4f} \n"
            # print(desc)
            # loop_obj.set_description(desc)

        stats = {"Train_loss" : total_train_loss,
                 "Train_accuracy" : total_train_accuracy,
                 "Validation_loss" : total_validation_loss,
                 "Validation_accuracy" : total_validation_accuracy}

        return stats

class Inference_Loop():
    def __init__(self, nn : NN, x_test):
        self.nn = nn
        self.x_test = x_test
    
    def batch_generator(self, X, batch_size : int):
        num_samples = X.shape[0]
        indices = np.arange(num_samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices]

    def make_inference(self, batch_size : int) -> npt.NDArray[np.float64]:
        y_test = []

        for X_batch in tqdm(self.batch_generator(self.x_test, batch_size), desc = f"inference-progress-bar"):

                # getting the output
                Y_pred = self.nn.forward(X_batch)

                # getting the total number of correct answers
                Y_pred_class = np.argmax(Y_pred, axis = 1)
                y_test.append(Y_pred_class)
        
        return np.concatenate(y_test, dtype = np.int32)