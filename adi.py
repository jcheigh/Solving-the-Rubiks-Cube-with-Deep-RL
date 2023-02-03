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
    for itr in tqdm(range(500)):
        cubes = get_scrambled_cubes(30, 30)

        X = get_training_data(cubes)  
        values, policies = get_values_and_policies(cubes, model) #Y in paper

        weights = get_sample_weights()

        model.fit(X, {"val" : values, "policy" : policies},
                  epochs = 15, sample_weight = weights,
                  verbose = 0)
        model.save(path)
    return model

def test_adi():
    model, _ = get_model()
    cube = Cube()
    nn_input = cube.get_nn_input()
    value, policy = model.predict(nn_input)
    print(f"Solved Cube Value: {value}")
    print(f"\nSolved Cube Policy: {policy}")
    cubes = get_scrambled_cubes(30, 30)
    values, policies = get_values_and_policies(cubes, model) #Y in paper
    print(np.argsort(values))
    print(np.argsort(policies))
    for i, value in enumerate(values):
        print(f"Value for cube {i+1}: {value}")
    for i, policy in enumerate(policies):
        print(f"Policy for cube {i+1}: {policy}")

if __name__ == "__main__":
    adi()