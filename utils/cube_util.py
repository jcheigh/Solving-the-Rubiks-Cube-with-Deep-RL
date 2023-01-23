from cube import Cube

def get_scrambled_cubes(batch_size, k):
    cubes = []
    for i in range(batch_size): 
        cube = Cube()
        cube.scramble(batch_size % k + 1)
        cubes.append(cube)
    return cubes