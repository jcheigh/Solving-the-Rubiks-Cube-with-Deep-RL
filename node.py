from cube import Cube
import numpy as np
from cube_util import get_cube_moves
from adi_util import get_model

class Node:
    #virtual loss hyperparameter
    virtual_loss = 150 
    #model
    model, _ = get_model()

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
        #parent Node 
        self.parent = None
        #children Nodes
        self.children = set()

    def is_leaf(self):
        return self.children == set()
        
    def set_num_took_action(self, x):
        self.num_took_action = x 
    
    def set_max_value_of_action(self, x):
        self.max_value_of_action = x

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

    def set_parent(self, parent):
        self.parent = parent 

    def get_parent(self):
        return self.parent 

    def add_children(self):
        cube = self.state 
        for successor in cube.get_successors():
            nn_input = successor.get_nn_input()
            _, policy = self.model.predict(nn_input)
            child_node = Node(successor, policy)
            child_node.set_parent(self)
            self.children.add(child_node)

    def get_children(self):
        return self.children