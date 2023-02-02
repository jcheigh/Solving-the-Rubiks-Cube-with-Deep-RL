from argparse import ArgumentParser 
from cube import Cube 
from utils.adi_util import *
from tensorflow import keras
from tqdm import tqdm
import tensorflow as tf
import os 
import pickle 

def adi():
    model, path = get_model() 
    for itr in tqdm(range(50)):
        cubes = get_scrambled_cubes(30, 30)

        X = get_training_data(cubes)  
        values, policies = get_values_and_policies(cubes, model) #Y in paper

        weights = get_sample_weights()

        model.fit(X, {"val" : values, "policy" : policies},
                  epochs = 15, sample_weight = weights,
                  verbose = 0)
        model.save(path)
    return model

if __name__ == "__main__":
    adi()