# Códigos de ML
import numpy as np
import random, os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seed(42)



def normalizaDados(data, zscore=True):
    for col in data.columns:
        if zscore:
            data.loc[:, col] = (data.loc[:, col] - data.loc[:, col].mean())/data.loc[:, col].std()
        else:
            minData, maxData = min(data[col]), max(data[col])
            if minData != maxData:
                data.loc[:, col] = (data.loc[:, col] - minData)/(maxData - minData)
            else:
                data.loc[:, col] = 1
    return


class NeuralNetwork(tf.keras.Model):


    def __init__(self, tamanhoVetor, lowerDim=3):
        super(NeuralNetwork, self).__init__()
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(lowerDim, activation='tanh'),
        ])
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(128, activation='elu'),
            tf.keras.layers.Dense(tamanhoVetor, activation='tanh')
        ])
        return
    

    def call(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y
    
    def encoded(self, x):
        return self.encoder(x)


    def customLoss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred)) + tf.reduce_mean((y_true - y_pred)**2)


class PCA():

    def __init__(self, data, components):
        data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
        self.mean = tf.reduce_mean(data_tensor, axis=0)
        self.centered_data = data_tensor - self.mean
        cov_matrix = tf.matmul(self.centered_data, self.centered_data, transpose_a=True) / tf.cast(tf.shape(data_tensor)[0] - 1, tf.float32)
        self.eigenvalues, self.eigenvectors = tf.linalg.eigh(cov_matrix)
        self.pca = PCA.sorting(self, components)


    def sorting(self, components):
        sorted_indices = tf.argsort(self.eigenvalues, direction='DESCENDING')
        sorted_eigenvectors = tf.gather(self.eigenvectors, sorted_indices, axis=1)
        sorted_eigenvalues = tf.gather(self.eigenvalues, sorted_indices)
        top_eigenvectors = sorted_eigenvectors[:, :components]
        pca_result = tf.matmul(self.centered_data, top_eigenvectors)
        return pca_result
