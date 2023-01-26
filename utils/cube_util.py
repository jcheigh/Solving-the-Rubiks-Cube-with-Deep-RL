import numpy as np

def get_basic_cube(face, dir):
    """
    Gets tuple for basic Cube associated with face/dir

    Parameters
    ----------
    face: string
          string f in {R,L,U,D,F,B} representing face to be turned
    dir: int
         integer d in {-1,1} representing CCW and CW rot, respectively
    Returns
    -------
    4-tuple of numpy arrays
    tuple representation of basic Cube
    
    """
    face = face if dir == 1 else face + "'" #R or R'

    #corner permutations for basic moves
    basic_cp = {'R':[1,2,7,3,5,6,8,4],'L':[5,1,3,4,6,2,7,8],
    'F':[4,2,3,8,1,6,7,5],'B':[1,6,2,4,5,7,3,8],'U':[2,3,4,1,5,6,7,8],
    'D':[1,2,3,4,8,5,6,7],"R'":[1,2,4,8,5,6,3,7],"F'":[5,2,3,1,8,6,7,4],
    "B'":[1,3,7,4,5,2,6,8],"L'":[2,6,3,4,1,5,7,8],"U'":[4,1,2,3,5,6,7,8],
    "D'":[1,2,3,4,6,7,8,5]}

    #edge permutations for basic moves
    basic_ep = {'R':[1,2,3,7,5,6,12,4,9,10,11,8],'L':[1,5,3,4,10,2,7,8,9,6,11,12],
    'F':[8,2,3,4,1,6,7,9,5,10,11,12],'B':[1,2,6,4,5,11,3,8,9,10,7,12],
    'U':[2,3,4,1,5,6,7,8,9,10,11,12],'D':[1,2,3,4,5,6,7,8,12,9,10,11],
    "R'":[1,2,3,8,5,6,4,12,9,10,11,7],"F'":[5,2,3,4,9,6,7,1,8,10,11,12],
    "B'":[1,2,7,4,5,3,11,8,9,10,6,12],"L'":[1,6,3,4,2,10,7,8,9,5,11,12],
    "U'":[4,1,2,3,5,6,7,8,9,10,11,12],"D'":[1,2,3,4,5,6,7,8,10,11,12,9]}

    #corner orientations for basic moves
    basic_co = {'R':[0,0,2,1,0,0,1,2],'L':[2,1,0,0,1,2,0,0],
    'F':[1,0,0,2,2,0,0,1],'B':[0,2,1,0,0,1,2,0],'U':[0,0,0,0,0,0,0,0],
    'D':[0,0,0,0,0,0,0,0],"R'":[0,0,2,1,0,0,1,2],"B'":[0,2,1,0,0,1,2,0],
    "F'":[1,0,0,2,2,0,0,1],"L'":[2,1,0,0,1,2,0,0],"U'":[0,0,0,0,0,0,0,0],
    "D'":[0,0,0,0,0,0,0,0]}

    #edge orientations for basic moves
    basic_eo = {'R':[0,0,0,0,0,0,0,0,0,0,0,0],'L':[0,0,0,0,0,0,0,0,0,0,0,0],
    'F':[1,0,0,0,1,0,0,1,1,0,0,0],'B':[0,0,1,0,0,1,1,0,0,0,1,0],
    'U':[0,0,0,0,0,0,0,0,0,0,0,0],'D':[0,0,0,0,0,0,0,0,0,0,0,0],
    "R'":[0,0,0,0,0,0,0,0,0,0,0,0],"L'":[0,0,0,0,0,0,0,0,0,0,0,0],
    "U'":[0,0,0,0,0,0,0,0,0,0,0,0],"D'":[0,0,0,0,0,0,0,0,0,0,0,0],
    "F'":[1,0,0,0,1,0,0,1,1,0,0,0],"B'":[0,0,1,0,0,1,1,0,0,0,1,0]}

    #wrapping in numpy array
    cp = np.array(basic_cp[face])
    ep = np.array(basic_ep[face])
    co = np.array(basic_co[face])[:-1]
    eo = np.array(basic_eo[face])[:-1]

    return (cp, ep, co, eo)

def sgn(perm):
    """
    Computes sgn(perm) := 1 iff perm is even, -1 o.w. in O(n)
    Code found online: credit to Derek O'Connor 20 March 2011
    We use face that sgn(perm) == (-1)^(# even length cycles)

    Parameters
    ----------
    perm: list/numpy array
          a permutation list (ex: [1,4,2,3])
    Returns
    -------
    int (+/- 1) == sgn(perm)
    """
    n = perm.size  
    visited = np.zeros(n) #checks visited 
    sgn = 1 
    for i in range(n):
        if not visited[i]: 
            next = i #start of new cycle
            len = 0
            while not visited[i]:
                len += 1
                visited[next] = 1 
                next = perm[next]
            if len % 2 == 0: #if even flip sign
                sgn *= -1
    return sgn