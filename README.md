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
- argparse, configparser, subprocess, zipfile, os.
  
In order to use Kaggle's API you may want to create authentication API token. Please refer to [Kaggle's docs](https://www.kaggle.com/docs/api).

## Usage
Create folder named 'data' in the project's directory. If you have downloaded dataset manually, store it there. Otherwise, leave it empty.

You can run the program in three ways:

Training and saving: `python MLP.py learn`

Loading and evaluating model: `python MLP.py load`

Tuning hyperparameters: `python MLP.py tune`

The default parameters are stored in the config.ini file. You can change them directly or run command-line arguments:
- `--layers`,
- `--epochs`,
- `--batch_size`,
- `--eta`,
- `--lmbda`,
- `--cost`,
- `--activation`.
  
Additionally several options are possible:
- `--show_history`,
-  `--visualize`,
-  `--download_data`,
-  `--n_trails`,
-  `--n_augmentations`.

To check what each options does, run `python MLP.py --help`.
  
Example: `python MLP.py --epochs 15 --batch_size 30`.

## Acknowledgements 
- Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015, [link](http://neuralnetworksanddeeplearning.com/index.html)
- Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019. Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD, [link](https://optuna.org/)
