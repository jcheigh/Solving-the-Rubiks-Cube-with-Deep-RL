from argparse import ArgumentParser 
from cube import Cube 
from utils.cube_util import get_scrambled_cubes
from utils.adi_util import get_model
from tensorflow import keras
import tensorflow as tf
import os 
import pickle 


#make path for target and output network
def main():
    model = get_model()
    for itr in range(10000):
        cubes = get_scrambled_cubes(1000, 30) #called X in paper
        for cube in cubes:
            
            for successor in cube.get_successors():
                nn_input = successor.get_nn_input()
                val, policy = model.predict(nn_input)
                #values.append(val + successor.get_reward())
                #policy.append()



            #for action in actions
                #value = value of best action + bellman
                #action = action of best action + bellman
            #train using (value, action)
            #update parameters 



        #get output then train
        #check for update

    #dump the network(s)
    return None 

if __name__ == "__main__":
    #main()
    """
    cube = Cube()
    cube.scramble(20)
    cube1 = Cube()
    model = ResNet()
    input = cube.get_nn_input()
    model.forward(input)
    #lst = [input, cube1.get_nn_input()]
    #model.forward(*lst)
    """