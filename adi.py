from argparse import ArgumentParser 
from cube import Cube 
from utils.adi_util import *
from tensorflow import keras
import tensorflow as tf
import os 
import pickle 

def adi():
    model, path = get_model()
    for itr in range(1000):
        cubes = get_scrambled_cubes(30, 30) #called X in paper
        X = get_training_data(cubes)  
        Y = [get_nn_output(cube, model) for cube in cubes]
        values = np.array([val for val, pol in Y])
        policies = np.array([pol for val, pol in Y])
        weights = get_sample_weights()
        model.fit(X, {"val" : values, "policy" : policies},
                  epochs = 15, sample_weight = weights)
        model.save(path)
    return model

if __name__ == "__main__":
    adi()