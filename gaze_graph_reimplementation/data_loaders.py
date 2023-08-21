import pandas as pd
import numpy as np
import math
from typing import Tuple, List
import tensorflow as tf
from tensorflow import keras
import random
import os
from keras.utils import to_categorical
from process_data import construct_matrices

def load_gaze_graph_csv(location):
    #TODO: doc strings
    df = pd.read_csv(location)
    df.columns = ['x', 'y']
    # print(df)
    zipped = list(zip(df.x, df.y))
    return zipped

def load_gg_dataset(data_location, class_to_num, num_classes = 6, k_hops = 2):
    #TODO: doc strings
    folders = [os.path.join(data_location, f) for f in os.listdir(data_location) if os.path.isdir(os.path.join(data_location, f))]
    folder_to_file_paths = dict()
    for folder in folders:
        csvs = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.csv')]
        folder_to_file_paths[folder] = csvs
    
    classes_to_dfs = dict()
    for i in range(num_classes):
        classes_to_dfs[i] = []
    
    for folder in folder_to_file_paths:
        for csv in folder_to_file_paths[folder]:
            class_name = str(csv).split('/')[-1].split('_')[-1].split('.')[0].lower()
            class_num = class_to_num[class_name]
            data = load_gaze_graph_csv(csv)
            matrix = construct_matrices(data, k_hops)
            matrix = np.resize(matrix, (90,90,2))
            # print(matrix.shape)
            classes_to_dfs[class_num].append(matrix)
    
    train_set = [[],[]]
    val_set = [[],[]]
    test_set = [[],[]]

    for category in classes_to_dfs:
        data_list = classes_to_dfs[category]
        n = len(data_list)
        rand_val = random.randint(0, n-1)
        val_set[0].append(data_list.pop(rand_val))
        val_set[1].append(category)
        rand_test = random.randint(0, n-2)
        test_set[0].append(data_list.pop(rand_test))
        test_set[1].append(category)
        train_set[0]+= data_list
        train_set[1]+= [category for i in data_list]
    
    train_set[0] = np.stack(train_set[0])
    train_set[1] = np.array(train_set[1])
    train_set[1] = to_categorical(train_set[1])
    val_set[0] = np.stack(val_set[0])
    val_set[1] = np.array(val_set[1])
    val_set[1] = to_categorical(val_set[1])
    test_set[0] = np.stack(test_set[0])
    test_set[1] = np.array(test_set[1])
    test_set[1] = to_categorical(test_set[1])

    train_dataset = tf.data.Dataset.from_tensor_slices((train_set[0], train_set[1]))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_set[0], test_set[1]))

    BATCH_SIZE = 2
    SHUFFLE_BUFFER_SIZE = 8

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset

def load_fluid_csv():
    pass