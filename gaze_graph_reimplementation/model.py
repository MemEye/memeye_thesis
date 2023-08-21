import pandas as pd
import numpy as np
import math
from typing import Tuple, List
import tensorflow as tf
from tensorflow import keras
import random
import os
from keras.utils import to_categorical
from data_loaders import load_gg_dataset
import yaml

def create_tf_model(num_classes, dropout_rate = 0.01):
    #using relu activation
    #reshape matrices to 90x90s
    conv1 = keras.layers.Conv2D(32, (5, 5), input_shape=(90, 90, 2), activation='relu')
    conv2 = keras.layers.Conv2D(32, (5, 5), activation='relu')
    pool1 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')
    # do we need to reshape? tf.image.resize_images(pool1, 32)
    dropout1 = keras.layers.Dropout(rate=dropout_rate)
    conv3 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation= 'relu')
    pool2 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')
    dropout2 = keras.layers.Dropout(rate=dropout_rate)
    conv4 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation= 'relu')
    pool3 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')
    dropout3 = keras.layers.Dropout(rate=dropout_rate)
    flatten = keras.layers.Flatten()
    fc1 = keras.layers.Dense(units=64)#, activation=tf.nn.sigmoid)
    fc2 = keras.layers.Dense(units=num_classes)#, activation=tf.nn.softmax)

    model = keras.Sequential([conv1, conv2, pool1, dropout1, 
                              conv3, pool2, dropout2, 
                              conv4, pool3, dropout3,
                              flatten, fc1, fc2])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
    
    return model

def train_model(model, train_dataset, epochs = 10):
    model.fit(train_dataset, epochs=epochs)

def run_pipeline(num_classes, sample_set, class_to_num):
    train_dataset, test_dataset = load_gg_dataset(sample_set, class_to_num)

    model = create_tf_model(num_classes= num_classes) #browse, play, read, search, watch, write

    train_model(model, train_dataset=train_dataset)


if __name__ == '__main__':

    with open('/Users/monaabd/Desktop/meng/memeye_thesis/gaze_graph_reimplementation/model_config.yaml') as config:
        config = yaml.safe_load(config)

    parameters = config['parameters']
    
    if parameters['gaze_graph_config']['gaze_graph_data']:
        class_to_num = parameters['gaze_graph_config']['class_to_num']
        data_loc = parameters['gaze_graph_config']['data_location']
    else:
        exit() #TODO: implement data loader for sams prev data

    num_classes = len(class_to_num)

    run_pipeline(num_classes, data_loc, class_to_num)

    



