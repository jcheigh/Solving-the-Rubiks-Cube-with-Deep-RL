from cube import Cube
from node import Node
import numpy as np
from adi_util import get_model
class MCTS():

    def __init__(self, cube, model):
        #model
        self.model = model 
        #cube
        self.cube = cube 
        #root node
        self.root = self.get_root_node()
        #exploration hyperparameter 
        self.exploration_param = 100

    def get_root_node(self):
        cube = self.cube 
        _, policy = self.model.predict(cube.get_nn_input())
        return Node(cube, policy)

    def choose_action(self, curr):
        N, W, L, P = curr.get_memory()
        Q = W - L
        u_cons = self.exploration_param * np.sqrt(np.sum(N))
        U = (u_cons * P) / (1 + N)
        action = np.argmax(U + Q)
        curr.update_vir_loss(action)
        return action

    def traverse(self):
        root = self.root
        path = [root]
        actions = []
        curr = root
        while curr.has_children():
            print(curr.state)
            if curr.is_solved_cube():
                return path, [], True
            action = self.choose_action(curr)
            actions.append(action)
            curr = curr.get_children()[action]
            path.append(curr)
        return path, actions, False

    def backpropagate(self, path, actions, value):
        for node, action in zip(path, actions):
            node.update_max_value_of_action(action, value)
        return None 

    def run_simulations(self, num_trials = 10000):      
        for trial in range(num_trials):
            path, actions, is_solved = self.traverse()
            if is_solved:
                return actions 
            leaf = path[-1]
            leaf_val, _ = leaf.add_children(self.model)
            actions.append(self.choose_action(leaf))
            self.backpropagate(path, actions, leaf_val)
        return None 


