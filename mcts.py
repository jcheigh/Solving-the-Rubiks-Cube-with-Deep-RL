from cube import Cube
import numpy as np
from cube_util import get_cube_moves
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
    
class MCTS():
    model, _ = get_model()

    def __init__(self, root):
        #root node
        self.root = root

        #current state 
        self.current = root

        #visited states 
        self.visited = {root}

    
    def search(self):
        return None
        #do MCTS search

    def select(self):
        return None
        #perform selection

    def expand(self):
        return None

    def backpropagate(self):
        return None 

