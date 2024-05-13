# Neural Network from Scratch
This project is dedicated to building a neural network from scratch to recognize digits. Its aim is to understand how neural networks function, including implementing methods to improve performance and manage overfitting.

## Features
- Activation Functions: sigmoid and ReLu,
- Cost Functions: Quadratic or Cross Entropy,
- Stochastic Gradient Descent optimizer,
- Early stopping,
- Regularization.

## Dataset
The data set used in this project is pupular MNIST data set which contains exaples of handwriting digits. The dataset was already provided in CSV format and can be downloaded, extracted and preprocessed within the code.

You can download data manually [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

## Prerequsities (Python libraries)
- numpy,
- pandas (only for reading csv).
  
Optional (for downloading and extracting data):
- kaggle, zipfile, os.
  
In order to use Kaggle's API you may want to create authentication API token. Please refer to [Kaggle's docs](https://www.kaggle.com/docs/api).

## Usage
0. If you have downloaded the dataset manually, please specify its path when callig `pd.read_csv()` for `train` and `test`. Do not remove `to_numpy()` method.
1. Create an instance of the `Network` class. Its inputs are:
    - list of number of neurons in each layer (e.g. `[10,30,5]` implies 3 layers with 10 input neurons, 30 hidden neurons and 10 output neurons. **Note that because of the nature of the dataset the input layer has to be of size 784** (28x28 pixels).
    - cost function: `'quadratic'` or `'cross_entropy'`.
    - activation function: `'sigmoid'` or `'ReLu'`.
 2. Call `fit()` method on the instance. Its inputs are:
   - a list, where first element is the training features and the second its lables,
   - batch size for SGD,
   - number of epochs,
   - learning rate,
   - lambda parameter for L2 regularization,
   - patience rate (10 by default)
   - a list, where first element is the validation features and the second its lables.

Example usage is available in MLP_recognizing_digits.ipynb.

## Bibliography
-Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015, [link](http://neuralnetworksanddeeplearning.com/index.html)
