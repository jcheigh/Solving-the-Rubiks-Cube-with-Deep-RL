from cube import Cube
from node import Node
import numpy as np
from cube_util import get_cube_moves
from adi_util import get_model
class MCTS():
    model, _ = get_model()

    def __init__(self, cube):
        #cube
        self.cube = cube 
        #root node
        self.root = self.get_root_node()
        #current node (maybe change up later)  
        self.current = self.root 

    def get_root_node(self):
        cube = self.cube 
        _, policy = self.model.predict(cube.get_nn_input())
        return Node(cube, policy)

    def choose_action(self):
        #choose action using formula in paper 
        #return an action in cube.moves
        return None 

    def traverse(self):
        root = self.get_root_node()

    def search(self, num_trials):
        for trial in range(num_trials):
            leaf = self.traverse()
            #not correct below but general idea
            self.expand(leaf)
            self.backpropagate(leaf)
        return None

    def expand(self):
        return None

    def backpropagate(self):
        return None 

