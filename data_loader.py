import numpy as np
import pandas as pd 
import subprocess
import zipfile
import os

#### Data download and preprocessing ####
class mnist_loader:
    def __init__(self, download=True, path='./data', n_augment=0, use_gpu=False):
        '''' 
        If download is True, download and extract to directory specified in path. 
        Otherwise, load the dataset from the directory specified in path. 
        '''
        self.path = path
        self.n_augment = n_augment
        if download:
            self.download_and_extract_data()
        try:
            self.train, self.valid, self.test = self.load_data()
        except Exception as e:
            print(f"Failed to load data: {e}. Reprocessing data...")
            self.train, self.valid, self.test = self.preprocess()

    def download_and_extract_data(self):
        # TO DO: Extract when already downloaded
        ''' Use Kaggle API to download and extract the MNIST dataset.'''
        dataset = 'oddrationale/mnist-in-csv'
        result = subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset, '-p', self.path], check=True)
        if result.returncode != 0:
            raise Exception('Failed to download the dataset. Make sure to have Kaggle API installed. Alternatively, download dataset manually: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv')
        else:
            zip_path = os.path.join(self.path, 'mnist-in-csv.zip')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.path)

    def preprocess(self):
        print('Preprocessing data...')
        training_data = pd.read_csv(os.path.join(self.path, 'mnist_train.csv')).values
        test_data = pd.read_csv(os.path.join(self.path, 'mnist_test.csv')).values
        X_train, Y_train = training_data[:,1:] / 255 , training_data[:,0]
        X_test, Y_test = test_data[:,1:] / 255, test_data[:,0]

        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm].T
        Y_train = Y_train[perm]

        X_valid = X_train[:,:10000]
        X_train = X_train[:,10000:]
        X_test=X_test.T

        Y_valid = Y_train[:10000]
        Y_train = Y_train[10000:]
        Y_train = np.eye(10)[Y_train].T
        Y_valid = np.eye(10)[Y_valid].T
        Y_test = np.eye(10)[Y_test].T
        
        if self.n_augment > 0:
            augmented_images, augmented_labels = self.augment_data(X_train, Y_train)
            X_train = np.concatenate((X_train, np.hstack(augmented_images)), axis=1)
            Y_train = np.concatenate((Y_train, np.hstack(augmented_labels)), axis=1)
        
        # Save the preprocessed arrays in the specified path
        np.save(os.path.join(self.path, 'X_train.npy'), X_train)
        np.save(os.path.join(self.path, 'Y_train.npy'), Y_train)
        np.save(os.path.join(self.path, 'X_valid.npy'), X_valid)
        np.save(os.path.join(self.path, 'Y_valid.npy'), Y_valid)
        np.save(os.path.join(self.path, 'X_test.npy'), X_test)
        np.save(os.path.join(self.path, 'Y_test.npy'), Y_test)

        train, valid, test = (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)
        print('Preprocessing complete.')
        return train, valid, test
        
    def augment_data(self, images, labels): # to do: augment list of images
        def bilinear_interpolate(image, x, y):
            x0, x1 = int(np.floor(x)), int(np.ceil(x))
            y0, y1 = int(np.floor(y)), int(np.ceil(y))
            if x0 < 0 or x1 >= image.shape[0] or y0 < 0 or y1 >= image.shape[1]:
                return 0
            Ia, Ib, Ic, Id = image[x0, y0], image[x0, y1], image[x1, y0], image[x1, y1]
            wa, wb, wc, wd = (x1 - x) * (y1 - y), (x1 - x) * (y - y0), (x - x0) * (y1 - y), (x - x0) * (y - y0)
            return Ia * wa + Ib * wb + Ic * wc + Id * wd

        def rotate_image(image, angle_range=(-15, 15)):
            angle = np.random.uniform(angle_range[0], angle_range[1])
            angle_rad = np.deg2rad(angle)
            A = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
            
            height, width = image.shape
            cx, cy = width // 2, height // 2
            rotated_image = np.zeros_like(image)
            
            for x in range(width):
                for y in range(height):
                    (x_new, y_new) = np.dot([x - cx, y - cy], A) + (cx, cy)
                    if 0 <= x_new < width and 0 <= y_new < height:
                        rotated_image[x, y] = bilinear_interpolate(image, x_new, y_new)   
            return rotated_image

        def shift_image(image, shift_range=(-6, 6)):
            shift_x = np.random.randint(shift_range[0], shift_range[1])
            shift_y = np.random.randint(shift_range[0], shift_range[1])
            
            height, width = image.shape
            shifted_image = np.zeros_like(image)
            
            for x in range(width):
                for y in range(height):
                    x_new = x + shift_x
                    y_new = y + shift_y
                    if 0 <= x_new < width and 0 <= y_new < height:
                        shifted_image[x, y] = image[x_new, y_new]
            return shifted_image
        augmented_images = []
        augmented_labels = []
        for i in range(self.n_augment):
            transform_func = np.random.choice([rotate_image, shift_image])
            original_image = images[:,i].reshape(28, 28)
            y=labels[:,i].reshape(-1,1)
            augmented_image = transform_func(original_image).reshape(784, 1)
            augmented_images.append(augmented_image)
            augmented_labels.append(y)
            print(f'Augmented {i+1}/{self.n_augment} images', end='\r')
        print()
        return augmented_images, augmented_labels
    
    def load_data(self): # to do: add augmentation
        X_train = np.load(os.path.join(self.path, 'X_train.npy'))
        Y_train = np.load(os.path.join(self.path, 'Y_train.npy'))
        X_valid = np.load(os.path.join(self.path, 'X_valid.npy'))
        Y_valid = np.load(os.path.join(self.path, 'Y_valid.npy'))
        X_test = np.load(os.path.join(self.path, 'X_test.npy'))
        Y_test = np.load(os.path.join(self.path, 'Y_test.npy'))

        train, valid, test = (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)
        return train, valid, test