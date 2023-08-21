import pandas as pd
import numpy as np
import math
from typing import Tuple, List
import tensorflow as tf
from tensorflow import keras
import random
import os
from keras.utils import to_categorical

def euclid_distance(next, curr):
    return math.sqrt((next[0] - curr[0])**2 + (next[1] - curr[1])**2)

def gaze_orientation(next, curr):
    try:
        angle = math.atan(((next[1] - curr[1])/(next[0] - curr[0])))
    except:
        try:
            x = (next[0] - curr[0])
            y = (next[1] - curr[1])
            sign = 1 if (x*y) > 0 else -1
            angle = -1 * sign * math.atan((abs(x)-abs(y))/(abs(x)+abs(y)))
        except:
            angle = 0

    
    return angle

def construct_matrices(nodes: List[float], k_hops: int):
    n = len(nodes)
    m_dist = np.zeros((n,n))
    m_orient = np.zeros((n,n))

    for i in range(0, n):
        for j in range(i-k_hops, i+k_hops):
            if j >= 0 and j < n:
                dist = euclid_distance(nodes[j], nodes[i])
                orient = gaze_orientation(nodes[j], nodes[i])
                m_dist[i, j] = dist
                m_orient[i, j] = orient

    return np.dstack((m_dist, m_orient))