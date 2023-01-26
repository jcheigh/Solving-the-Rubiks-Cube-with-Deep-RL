import os 
from model import make_model, compile_model
from tensorflow import keras
from cube import Cube

def get_model():
    path = "/Users/jcheigh/ML-Projects/Solving the Rubik's Cube with Deep RL/Models/model"
    model = None
    if os.path.exists(path):
        model = keras.models.load_model(path)
    else:
        model = make_model()
        compile_model(model)
    return model

def get_scrambled_cubes(batch_size, k):
    cubes = []
    for i in range(batch_size): 
        cube = Cube()
        cube.scramble(batch_size % k + 1)
        cubes.append(cube)
    return cubes