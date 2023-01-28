import os 
import numpy as np
from model import make_model, compile_model
from tensorflow import keras
from cube import Cube

def get_model():
    path = "/Users/jcheigh/Solving-the-Rubik-s-Cube-with-Deep-RL/saved_models"
    model = None
    if os.path.exists(path):
        model = keras.models.load_model(path)
    else:
        model = make_model()
        compile_model(model)
    return model, path

def get_scrambled_cubes(batch_size, k):
    cubes = []
    for i in range(batch_size): 
        cube = Cube()
        cube.scramble(batch_size % k + 1)
        cubes.append(cube)
    return cubes

def get_nn_output(cube, model):
    successors = cube.get_successors()
    value, policy = -np.inf, ''
    for move, successor in successors:
        nn_input = successor.get_nn_input()
        succ_val, succ_policy = model.predict(nn_input)
        print(f"Successor Value: {succ_val}\n")
        print(f"Successor Policy: {succ_policy}\n")
        val = succ_val + successor.get_reward()
        if val > value:
            value, policy = val, move
    return value, policy

def get_training_data(cubes):
    X = np.empty((30, 480))
    for i, cube in enumerate(cubes):
        X[i] = np.array(cube.get_nn_input())
    return X 

def get_sample_weights():
    return np.array([1/(i+1) for i in range(30)])#np.array([1/(i+1) for i in range(30)] for j in range(1000)).flatten()


