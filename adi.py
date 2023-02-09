from argparse import ArgumentParser 
from cube import Cube 
from adi_util import *
from tensorflow import keras
from tqdm import tqdm
import tensorflow as tf
import os 
import pickle 

def adi():
    model, path = get_model() 
    for itr in tqdm(range(200)):
        cubes = get_scrambled_cubes(30, 30)

        X = get_training_data(cubes)  
        values, policies = get_values_and_policies(cubes, model) #Y in paper

        weights = get_sample_weights()

        model.fit(X, {"val" : values, "policy" : policies},
                  epochs = 15, sample_weight = weights,
                  verbose = 0)
        model.save(path)
    return model

def test():
    model, _ = get_model()
    cubes = get_scrambled_cubes(30, 30)
    values, policies = get_values_and_policies(cubes, model) #Y in paper
    values = [val[0] for val in values]
    for i, value in enumerate(values):
        print(f"Value for cube {i+1}: {value}")
    for i, policy in enumerate(policies):
        print(f"Policy for cube {i+1}: {policy}")
    print(np.argsort(values))

if __name__ == "__main__":
    test()