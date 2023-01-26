from argparse import ArgumentParser 
from cube import Cube 
from utils.adi_util import get_model, get_scrambled_cubes
from tensorflow import keras
import tensorflow as tf
import os 
import pickle 

def adi():
    model = get_model()
    for itr in range(10000):
        cubes = get_scrambled_cubes(1000, 30) #called X in paper
        for cube in cubes:
            move_dict = cube.get_successors()
            successor_values = dict()
            for move in move_dict:
                successor = move_dict[move]
                nn_input = successor.get_nn_input()
                val, policy = model.predict(nn_input)
                value = val + successor.get_reward()
                successor_values[move] = value
            cube_value = max(successor_values.items)
            cube_policy = list(move_dict.values()).index(cube_value)   
            to_train = (cube_value, cube_policy)
            #update parameters 
        #get output then train
        #check for update
    #dump the network(s)
    return None 

if __name__ == "__main__":
    print("Hello World")
    #main()