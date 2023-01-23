from argparse import ArgumentParser 
from cube import Cube
from model import ResNet
from utils.cube_util import get_scrambled_cubes
from utils.avi_util import get_models
import os 
import torch 
import pickle 

#make path for target and output network
def main():
    current_model, target_model = get_models()
    for itr in range(10000):
        cubes = get_scrambled_cubes(1000, 30) #called X in paper
        for cube in cubes:
            #train
            nn_input = cube.get_nn_input()



        #get output then train
        #check for update

    #dump the network(s)
    return None 

if __name__ == "__main__":
    #main()
    cube = Cube()
    cube.scramble(20)
    cube1 = Cube()
    model = ResNet()
    input = cube.get_nn_input()
    model.forward(input)
    #lst = [input, cube1.get_nn_input()]
    #model.forward(*lst)