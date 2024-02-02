#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
import os
    
def read_images_labels(images_filepath, labels_filepath):        
    labels = []
    with open(os.path.join(os.path.dirname(__file__), labels_filepath), 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
    
    with open(os.path.join(os.path.dirname(__file__), images_filepath), 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img            
    
    return np.array(images), np.array(labels)
        
def load_data():
    x_train, y_train = read_images_labels('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    x_test, y_test = read_images_labels('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    return (x_train, y_train),(x_test, y_test)      