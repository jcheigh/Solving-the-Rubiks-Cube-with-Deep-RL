from adi import adi 
from mcts import MCTS 
from adi_util import get_model
from cube import Cube

def solve(cube, pretrained = True):
    if cube.is_solved():
        print("The Cube is already solved!")
        return []
    if not pretrained:
        model = adi()
    else:
        model, _ = get_model()
    solver = MCTS(cube, model)
    move_dict = cube.get_moves()
    solution = solver.run_simulations()
    if solution is None:
        print(f"Sorry, simulation failed. Maybe try run_simulations(20000)")
    else:
        moves_to_solve = [move_dict[index] for index in solution]
        return moves_to_solve

if __name__ == "__main__":
    cube = Cube()
    cube.scramble(10)
    solve(cube)