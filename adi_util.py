import os 
import numpy as np
from model import make_model, compile_model
from tensorflow import keras
from cube import Cube

def get_model():
    path = "/Users/jcheigh/saved_models"
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
        cube.scramble(i % k + 1)
        cubes.append(cube)
    return cubes

def one_hot(i):
    result = np.zeros(12)
    result[i] = 1
    return result.reshape(1, -1)

def get_nn_output(cube, model):
    successors = cube.get_successors()
    value, policy = -np.inf, ''
    for move, successor in enumerate(successors):
        nn_input = successor.get_nn_input()
        succ_val, _ = model.predict(nn_input, verbose = 0)
        val = succ_val + successor.get_reward()
        if val > value:
            value, policy = val, move 
    return value, one_hot(policy)

def get_values_and_policies(cubes, model):
    values, policies = [], []
    for cube in cubes:
        val, pol = get_nn_output(cube, model)
        values.append(val)
        policies.append(pol)
    values = np.array(values).reshape(30, -1)
    policies = np.array(policies).reshape(30,-1)
    return values, policies 

def get_training_data(cubes, n = 30):
    X = np.empty((n, 480))
    for i, cube in enumerate(cubes):
        X[i] = np.array(cube.get_nn_input())
    return X 

def get_sample_weights():
    return np.array([1/(i+1) for i in range(30)])
    #np.array([1/(i+1) for i in range(30)] for j in range(1000)).flatten()

def test():
    print(f"Testing get_scrambled_cubes: \n")
    cubes = get_scrambled_cubes(30, 30)
    print(f"Scrambled Cubes: {cubes}")
    
    print(f"Testing one_hot")
    assert one_hot(5).shape == (1,12)
    print(one_hot(5))
    
    print(f"Testing get_nn_output")
    cube = Cube()
    cube.scramble(10)

    cube1 = Cube()
    cube1.scramble(10)

    model = make_model()
    compile_model(model)

    value, policy = get_nn_output(cube, model)
    print(f"Value: {value}")
    print(f"Policy: {policy}")

    print(f"Testing get_values_and_policies")
    cubes = get_scrambled_cubes(30,30)
    values, policies = get_values_and_policies(cubes, model)
    print(f"Values: {values}")
    print(f"Policies: {policies}")

    print(f"Testing get_training_data") 
    print(get_training_data(cubes))
    print(get_training_data(cubes)[0].reshape(20,24))

    print(f"Testing get_sample_weights") 
    print(get_sample_weights())

if __name__ == "__main__":
    test()



