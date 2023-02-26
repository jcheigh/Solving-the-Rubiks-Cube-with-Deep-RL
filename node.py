from cube import Cube
import numpy as np
from adi_util import get_model

class Node:
    #virtual loss hyperparameter
    virtual_loss = 150 

    def __init__(self, state, policy):
        #state is Cube(), policy is output of nn = np.array with .shape = (1,12)

        self.state = state 
        #N_s(a) = number of times took action a at state self.state
        self.num_took_action = np.zeros(12) 
        #W_s(a) = max value of action a at state self.state
        self.max_value_of_action = np.zeros(12)
        #L_s(a) = virtual loss for action a at state self.state 
        self.vir_loss_of_action = np.zeros(12)
        #P_s(a) = prior prob distribution of actions given self.state
        self.prob_of_action = policy 
        #Memory is tuple (N,W,L,P)
        self.memory = self.get_memory()
        #children Nodes
        self.children = []

    def update_vir_loss(self, index):
        self.vir_loss_of_action[index] +=  self.virtual_loss

    def has_children(self):
        return self.children != []

    def is_solved_cube(self):
        return self.state.is_solved()

    def is_leaf(self):
        return self.children == set()
        
    def update_num_took_action(self, index):
        self.num_took_action[index] += 1
    
    def update_max_value_of_action(self, index, value):
        if value > self.max_value_of_action[index]:
            self.max_value_of_action[index] = value 

    def set_vir_loss_of_action(self, x):
        self.vir_loss_of_action = x 
    
    def prob_of_action(self, x):
        self.prob_of_action = x

    def get_state(self):
        return self.state 
    
    def get_memory(self):
        # (N_s(a), W_s(a), L_s(a), P_s(a))
        return (
               self.num_took_action,
               self.max_value_of_action,
               self.vir_loss_of_action,
               self.prob_of_action 
        )   

    def predict(self, cube, model):
        nn_input = cube.get_nn_input()
        return model.predict(nn_input)  

    def add_children(self, model):
        cube = self.state 
        for successor in cube.get_successors():
            _, policy = self.predict(successor, model)
            child_node = Node(successor, policy)
            self.children.append(child_node)
        return self.predict(cube, model)

    def get_children(self):
        return self.children