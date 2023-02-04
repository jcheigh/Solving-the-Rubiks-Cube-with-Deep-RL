import numpy as np
import random 
from cube_util import get_basic_cube, sgn
import tensorflow as tf
class Cube:
    """
    Rubik's Cube class that holds the Cube as a 4-tuple (explained below). 
    Supports basic moves and the group operation of the Rubik's Cube (combining 2 configurations)

    cube = Cube((cp, ep, co, eo)) or cube = Cube() #defaults to solved Cube

    See writeup attached to README for more information. Essentially, the state of the Cube is fully determined 
    by the orientation and permutation of the corners/edges. We arbitrarily assign 1-8 & 1-12 for the 
    spatial positions of corners and edges. 

    Parameters
    ----------
    cp: numpy array of length 8 (element of S8)
        array representing permutation of corners 
        ex: np.array([2,3,6,8,7,1,5,4]) tells us corner i is in position arr[i]
    ep: numpy array of length 12 (element of S12)
        array representing permutation of edges (an element of 12)
        ex: np.array([1,2,3,4,5,6,7,8,9,10,12,11]) tells us edge 11 flips w/ edge 12
    co: numpy array of length 7 (element of (Z_3)^7)
        array representing orientation of corners 
        ex: the ith element of np.array([0,1,2,2,1,0,1]) is the orientation of the corner in the ith 
        spatial position
    eo: numpy array of length 11 (element of (Z_2)^{11})
        array representing orientation of edges 
        ex: the ith element of np.array([0,0,0,0,0,0,0,0,0,0,1]) is the orientation of the edge in
        the ith spatial position

    The reason why we only specify the first 7 corners and the first 11 edges is because the orientation
    of the last one is predetermined (see the Fundamental Theorem of Cubology for more details).

    There are 3 representations we use in this class:
        1) self.rep is the 4-tuple specified above
        2) self.state is the 4-tuple with the orientation of the last corner/edge specified 
        3) self.nn_input is a (1,480) one-hot tensor that we use as input for our neural network
    """
    #faces of the Cube according to Singmaster notation
    faces = ["R", "L", "U", "D", "F", "B"] 

    #we specify a CW turn of the R face by (R,1), and CCW with (R,-1)
    moves = [(f,d) for f in faces for d in [-1,1]] 

    #self.rep for the solved Rubik's Cube
    solved_rep = (np.arange(1,9), np.arange(1,13), np.zeros(7), np.zeros(11))

    def __init__(self, rep = solved_rep):
        """
        Constructor for Rubik's Cube. Takes as input a representation of the Cube in form of self.rep
        """
        #tuple representation
        self.rep = rep 

        #full tuple representation
        self.state = self.to_state() 

        #neural network input
        self.nn_input = self.to_nn_input() 

        #bool specifying if state is solvable
        self.valid = self.is_valid() 
    
    def copy(self):
        rep = self.rep
        return Cube(rep)

    def to_state(self):
        """
        Converts from tuple representation to full tuple representation using Fundamental Thm of Cubology.
        i.e. (S8,S12,(Z_3)^7,(Z_2)^{11}) => (S8,S12,(Z_3)^8,(Z_2)^{12})

        Returns
        -------
        4-tuple of numpy arrays
        Full tuple representation of self
        """
        #permutation/orientation of corners/edges
        c_perm, e_perm, c_orient, e_orient = self.rep 

        #get orientation of corners/edges necessary to be self.valid
        last_corner = (-1 * np.sum(c_orient)) % 3 
        last_edge = (-1 * np.sum(e_orient)) % 2

        #update c_perm, e_perm
        c_orient = np.append(c_orient, last_corner)
        e_orient = np.append(e_orient, last_edge) 

        #return full tuple representation
        return (c_perm, e_perm, c_orient, e_orient)

    def to_nn_input(self):
        """
        Converts from full tuple representation to neural network input (20x24 matrix). The first 8 rows
        are one-hot encoded locations of corners, next 12 are for edges. 

        Returns
        -------
        np.array with np.shape(arr) = (20, 24)
        Representation of self for neural network input


        Corners: ith col == 1 means the corner is in i//8th orientation and (i % 8) + 1 spatial position
        Edges: ith col == 1 means edge is in i//12 orientation and (i % 12) + 1 spatial position
        """
        #permutation/orientation of corners/edges
        c_perm, e_perm, c_orient, e_orient = self.state 

        #to return (20x24 matrix of zeros)
        result = np.zeros((20,24)) 

        #one hot encode positions 
        for row in range(20): 
            if row < 8: #corners
                pos, orient = c_perm[row], c_orient[c_perm[row] - 1] + 1 #double index due to convention 
            else: #edges
                pos, orient = e_perm[row - 8], e_orient[e_perm[row - 8] - 1] + 1 #+1 avoids certain issues 
            column = pos * orient - 1
            result[row, int(column)] = 1

        #flatten/expand dims
        result = np.expand_dims(result.flatten(), 0)

        #convert to tensor and return
        return tf.convert_to_tensor(result, dtype = tf.float32)

    def is_valid(self):
        """
        Checks if a state is valid. By the Fundamental Thm of Cubology it suffices to show
        sgn(corner_permutation) == sgn(edge_permutation)

        Returns
        -------
        Boolean
        True iff state is valid (i.e. solvable using sequence of basic moves)
        """
        corner_perm, edge_perm, _, _ = self.state 
        return sgn(corner_perm) == sgn(edge_perm)

    def get_reward(self):
        """
        Gets reward associated with state of the Cube. 

        Returns
        -------
        int
        1 iff Cube is solved else -1
        """
        return 1 if self.is_solved() else -1

    def get_successors(self):
        """
        Returns 
        -------
        List[Cube] 
        List containing Cubes obtained by performing each of the 12 basic moves
        """
        return [self.move(face, dir, in_place = False) for face, dir in self.moves]

    def get_state(self):
        """
        Returns
        -------
        Tuple of np.array
        Getter for self.state
        """
        return self.state 
    
    def get_nn_input(self):
        """
        Returns
        -------
        tf.tensor with shape (1,480)
        Getter for self.nn_input
        """
        return self.nn_input 
    
    def scramble(self, k):
        """
        Applies k random moves to the Cube. This is in place

        Returns
        --------
        None 
        In place operation
        """
        moves = random.choices(self.moves, k = k) #n random moves
        for (face, dir) in moves:
            self.move(face, dir)
        return None

    def combine(self, cube, in_place = True):    
        """
        Computes self * cube, which is the Cube resulting from applying the configuration cube to self.
        This is the standard group theory operation for the Rubik's Group

        Parameters
        ----------
        cube: Cube
              cube to be combined with self (note combine is not commutative)
        Returns
        -------
        Cube (if not in place otherwise returns None)
        Cube resulting from self * cube
        
        Let self.state := (a1,a2,a3,a4), cube.state := (b1,b2,b3,b4), self * cube := (c1,c2,c3,c4), then
            c1 = b1 composed a1, 
            c2 = b2 composed a2
            c3 = (a3 composed inverse(b1) + b3) mod 3
            c4 = (a4 composed inverse(b2) + b4) mod 2
        See writeup for more details. 
        """
        #get a_i's, b_i's
        a1, a2, a3, a4 = self.state 
        b1, b2, b3, b4 = cube.state 
        a1, a2, b1, b2 = a1 - 1, a2 - 1, b1 - 1, b2 - 1

        #quick way to perform composition
        c1 = b1[a1] + 1 
        c2 = b2[a2] + 1

        #nverse of permutation
        inv_b1 = np.argsort(b1)
        inv_b2 = np.argsort(b2)

        #get c3, c4 by formulas above
        c3 = (a3[inv_b1] + b3) % 3
        c4 = (a4[inv_b2] + b4) % 2

        #if in place or not
        if in_place:
            self.rep = (c1, c2, c3[:-1], c4[:-1])
            self.state = (c1, c2, c3, c4)
            self.nn_input = self.to_nn_input()
        else:
            return Cube((c1, c2, c3[:-1], c4[:-1]))
        return None

    def move(self, face, dir, in_place = True):
        """
        Turns the face of the Cube (dir == 1 is CW, -1 is CCW) by combining with basic Cubes.

        Parameters
        ----------
        face: string
              string f in {R,L,U,D,F,B} representing face to be turned
        dir: int
              integer d in {-1,1} representing CCW and CW rot, respectively
        Returns
        -------
        Cube (if not in place otherwise None)
        Cube resulting from making the correct rotation of the Cube
        """
        basic_cube = get_basic_cube(face, dir)
        if in_place:
            self.combine(Cube(basic_cube))
        else:
            return self.combine(Cube(basic_cube), in_place = False)

        return None 

    def __eq__(self, cube):
        """
        Returns
        -------
        Boolean
        True iff configurations of Cube are identical
        """
        return np.all(self.nn_input == cube.nn_input)

    def is_solved(self):
        """
        Returns
        -------
        Boolean
        True iff Cube is solved
        """
        return self.__eq__(Cube())

def test():
    # basic tests
    cube = Cube()
    assert cube.is_solved() and cube.valid

    for face, _ in cube.moves:
        # (face, -1) & (face, 1) are inverses
        cube.move(face, 1)
        cube.move(face, -1)
        assert cube.is_solved()

        # 4x any move solves 
        for i in range(4):
            cube.move(face, 1)
        assert cube.is_solved()

        for i in range(4):
            cube.move(face, -1)
        assert cube.is_solved()

    # Sune 6x should solve Cube
    for i in range(6):
        cube.move("R",1)
        cube.move("U",1)
        cube.move("R",-1)
        cube.move("U",-1)
    assert cube.is_solved()

    # Superflip 
    cube.move("R", -1)
    cube.move("U", 1)
    cube.move("U", 1)
    cube.move("B", 1)
    cube.move("L", -1)
    cube.move("F", 1)
    cube.move("U", -1)
    cube.move("B",1)
    cube.move("D", 1)
    cube.move("F", 1)
    cube.move("U",1)
    cube.move("D",-1)
    cube.move("L",1)
    cube.move("D", 1)
    cube.move("D", 1)
    cube.move("F",-1)
    cube.move("R",1)
    cube.move("B",-1)
    cube.move("D",1)
    cube.move("F",-1)
    cube.move("U",-1)
    cube.move("B",-1)
    cube.move("U",1)
    cube.move("D",-1)

    superflip_state = cube.state
    
    # Superflip is solved state with all misoriented eges
    assert all(superflip_state[0] == np.arange(1,9))
    assert all(superflip_state[1] == np.arange(1,13))
    assert all(superflip_state[2] == np.zeros(8))
    assert all(superflip_state[3] == np.ones(12))

    cube = Cube() 
    # R, U2, D', B, D' is an element with order 1260 
    for i in range(1260):
        cube.move("R",1)
        cube.move("U",1)
        cube.move("U",1)
        cube.move("D",-1)
        cube.move("B",1)
        cube.move("D",-1)    
    assert cube.is_solved()

    # basic moves are the successors of solved Cube
    basic_moves = [Cube(get_basic_cube(face, dir)) for (face, dir) in cube.moves]
    assert basic_moves == cube.get_successors()

    #scramble 100 times
    cube.scramble(100)
    #all valid
    assert np.all([c.valid for c in cube.get_successors()])
    
    #assert one hot encoded 
    assert np.sum(cube.nn_input) == 20

if __name__ == "__main__":
    test()

 

 


