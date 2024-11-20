import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import os

def load_data(file_path: str)->Tuple[np.ndarray, np.ndarray]:
    '''
    This function loads and parses text file separated by a ',' character and
    returns a data set as two arrays, an array of features, and an array of labels.
    Parameters
    ----------
    file_path : str
                path to the file containing the data set
    Returns
    -------
    features : ndarray
                1D array of shape (m) containing features for the data set
    labels : ndarray
                1D array of shape (n) containing labels for the data set -- n happens to be 1 here
    '''
    D = np.genfromtxt(file_path, delimiter=",")
    features = D[:, :-1]  # all columns but the last one
    labels = D[:, -1]  # the last column
    return features, labels

def initialize(weights_size):
    '''
    This method initializes the weight array and bias term at 0
    '''

    w = None # dummy value
    b = None # dummy value

    # TODO initialize the weight array and the bias term. The weight array should be weights_size long 
    # the bias term should just be a single value
    # Use must use numpy to initialize them. HINT: Look up numpy.zeros


    return w, b

def predict(x: np.ndarray, w: np.ndarray, b: np.ndarray)->np.ndarray:
    '''
    This method calculates the predicted value of property with attributes x, using weight array w, and bias b. 

    Parameters
    ----------
    X : np.ndarray
        An array of shape (m) where each row is a different set of features
    w : np.ndarray
        An array of shape (m) containing the weights for the linear model
    b : np.ndarray
        The bias term of shape (1)  
    Returns
    -------
    predictions : np.ndarray
        A scalar that is the prediction of value based off of the w and b provided
    '''

    prediction = None # Dummy value 

    # TODO Use the w and b vectors provided to make a prediction of the value of the input
    # Use the numpy library to do this using matrix notation (such as dot product) -- 
    # You can find docs here: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
 

    return prediction
    
def Mean_square_error(x, y, w, b):
    '''
    This method calculates the loss given a prediction and a true output

    Parameters
    ----------
    X : np.ndarray
        An array of shape (m) where each row is a different set of features
    w : np.ndarray
        An array of shape (m) containing the weights for the linear model
    b : np.ndarray
        The bias term of shape (1) 
    y    : np.ndarray
        The true vale of the same area  
    Returns
    -------
    loss : np.ndarray
        The MSE (as defined in the pdf) of the two
    '''

    loss = None # Dummy value 

    # TODO Calculate the Mean Squared Error (see the written section for the formula)
    # Hint: you may want to use a function you already wrote 

    return loss
    
def loss_gradient(x, y, w, b, pred):
    '''
    This method calculates the gradient of loss with respect to both the weight vector and the bias

    (I.E dL/dW and dL/db)

    Parameters
    ----------
    x : np.ndarray
        An array of shape (m) where each row is a different set of features
    y : np.ndarray
        The true value of the input    
    w : np.ndarray
        An array of shape (m) containing the weights for the linear model
    b : np.ndarray
        The bias term of shape (1) 
    pred : np.ndarray
        The prediction made with this w and b on input x 
    Returns
    -------
    d_w : np.ndarray
        The derivative of Loss with respect to the weight array
    d_b : np.ndarray
        The derivative of Loss with respect to the bias
    '''
    
    d_w = None # Dummy value 
    d_b = None # Dummy value 

    # TODO calculate the derivative of loss with respect to both the weights, w, and the bias b
    # hint : the length of the derivative is the same as the array itself (i.e. len(d_w) == len(w))
    # hint #2 : you already did all the math in part 1 (unless you skipped it cause math scares you)
 

    return d_w, d_b

def update(x, y, w, b, learning_rate):
    '''
    This method will update the weights and bias provided using a single training example

    This should use 'predict' and 'loss_gradient' to do this

    Parameters
    ----------
    x : np.ndarray
        An array of shape (m) where each row is a different set of features
    y : np.ndarray
        The true value of the input    
    w : np.ndarray
        An array of shape (m) containing the weights for the linear model
    b : np.ndarray
        The bias term of shape (1) 
    learning_rate : scalar
        The amount that we scale the gradient when updating 
    Returns
    -------
    new_w : np.ndarray
        The value of the weight array after updating
    new_b : np.ndarray
        The value of the bias after updating
    '''

    new_w = None # Dummy value 
    new_b = None # Dummy value 

    # TODO implement gradient descent -- that is assign new value for w and b
    # Recall that an iteration of SGD does the following: w = w - lr * dL/dw ; b = b - lr * dL/db


    return new_w, new_b

def grad_descent(w, b, data_input_list, data_label_list, learning_rate):
    '''
    This method does the ML -- it will iterate over the data and update the w and b each time while recording the loss at each step

    It should call 'Mean_squared_error' and 'update' methods that you wrote
    '''


    losses = []
    for num in range(len(data_input_list)):
        x = data_input_list[num]
        y = data_label_list[num]
        # TODO iteratively (in this loop) update w and b based off of the x and y above

        # TODO Record the loss for each datapoint in the losses array (so we can look at it work!)

    # Make sure to return the w and b that have been updated from training
    return w, b, losses

def plot_line(w, b, x, y, feat):
    '''
    Will create a plot showing the data (with only one input feature) 
    and the line of best fit represented by your w and b 
    no need to change this
    '''
    plt.scatter(x, y)
    plt.plot(x, w*x + b, 'r', linewidth=5)
    plt.title("Line of best fit for the " + str(feat) +"-th feature")
    plt.xlabel("setting of feature "+ str(feat))
    plt.ylabel("The actual value of the property (thousands of dollars)")
    plt.savefig("plots/line_of_best_fit_feat_"+str(feat)+".png")
    plt.show()


def learning_curve(losses):
    '''
    This will automatically plot your learning curve (loss vs training examples)
    '''

    # This averages the losses so that you don't get a super spiky learning curve
    losses = np.array(losses)
    losses = np.mean(losses.reshape(-1,22), axis=1)

    plt.plot(np.arange(len(losses))*23, losses)
    plt.axhline(y=50, color='red', linestyle='--')
    plt.ylim(0)
    plt.title("Training Curve for your model using Gradient Descent")
    plt.xlabel("Number of training steps")
    plt.ylabel("Loss")
    plt.savefig("plots/Learning_curve.png")
    plt.show()
    return


def main():
    '''
    This will call your methods to run the training and plotting 
    no need to change any of this -- unless it turns out I have a bug
    '''

    if not os.path.exists('plots'):
        os.makedirs('plots')

    X, Y = load_data("housing.csv")  # load the data set

    # This is normalizing the input, don't touch unless you want to make this homework a lot harder
    maxi = np.amax(X, axis=0)
    mini = np.amin(X, axis=0)
    X = (X-mini)/(maxi-mini)


    # TODO Find a learning rate that lets your model train below the threshold
    # It might be wise to make a for loop (none of the code in the main function is tested so go wild -- as long as you're making plots still)
    lr = 0.00001

    # TODO Uncomment the commented code (insie the ##s) when you reach the written question that asks for regression on each of the dimlensions
    # Hot tip: Most IDEs let you comment/uncomment block with ctrl+/

    # This loop trains trains the model 13 different times each on only 1 feature of the data and
    # plots the line of best fit each time 
    ##############################################################################
    # for i in range(13):
    #     x = X[:, [i]] # only loads the i-th feature of the inputs as the input
    #     w,b = initialize(1)
    #     w,b,_ = grad_descent(w, b, x, Y, lr)
    #     plot_line(w, b, x, Y, i)
    ##############################################################################



    # This bit will train on the full data and plot the training curve
    w,b = initialize(13)
    w,b,losses = grad_descent(w, b, X, Y, lr)
    learning_curve(losses)

    print("final loss with lr =", lr, ":", losses[-1])


if __name__ == "__main__":
    main()