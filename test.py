from genetic_problems import *
from genetic_algorithm import *
import numpy as np
from load_sudoku import *
import os
from glob import glob 

cwd = os.getcwd()

puzzlesPath = os.path.join(cwd, "puzzles/")
defaultPuzzleExtension = ".in"
puzzles = glob(puzzlesPath + "*" + defaultPuzzleExtension)
initials = []
for puzzle in puzzles:
    initials.append(load_puzzle(puzzle))

# print(len(initials))
# for initial in initials:
#     print(initial)
#     print(len(initial))

solutionsPath = os.path.join(cwd, "solutions/")
defaultSolutionExtension = ".sol"
sols = glob(solutionsPath + "*" + defaultSolutionExtension)
solutions = []
for sol in sols:
    solutions.append(load_solution(sol))

# print(len(solutions))
# for solution in solutions:
#     print(solution)
#     print(len(solution))


problem = SudokuGeneticProblem(initials[0])
genetic_algorithm(problem)