# Neural Network from Scratch
This project is dedicated to building a neural network from scratch to recognize digits. Its aim is to understand how neural networks function, including implementing methods to improve performance and manage overfitting.

## Features
- Activation Functions: sigmoid, ReLu,
- Cost Functions: Quadratic, Cross Entropy,
- Optimizer: Stochastic Gradient Descent,
- Early stopping,
- Regularization,
- Data augmentation (rotation and shifting),
- Hyperparameter tuning.

## Dataset
The data set used in this project is pupular MNIST data set which contains exaples of handwriting digits. The dataset was already provided in CSV format and can be downloaded, extracted and preprocessed within the code.

You can download the data manually [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

## Prerequsities
Third party:
- numpy,
- pandas,
- matplotlib,
- optuna (if you want to tune the hyperparameters),
- kaggle (if you want to download the data within the code).
  
Inbuilt:
- subprocess, zipfile, os.
  
In order to use Kaggle's API you may want to create authentication API token. Please refer to [Kaggle's docs](https://www.kaggle.com/docs/api).

## Usage
Run `python MLP.py` in the project's directory. The default parameters are set, but you can change them :
- `--epochs`,
- `--batch_size`,
- `--eta`,
- `--lmbda`.
  
Example: `python MLP.py --epochs 15 --batch_size 30`.

The rest of the parameters (for example the number of layers) need to be adjusted manually in the code.

## Acknowledgements 
- Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015, [link](http://neuralnetworksanddeeplearning.com/index.html)
- Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019. Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD, [link](https://optuna.org/)
