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
filename_initial_dict = {}
for puzzle in puzzles:
    basename = os.path.basename(puzzle)
    filename = os.path.splitext(basename)[0]
    filename_initial_dict.update( {
            filename: load_puzzle(puzzle)
        }
    )

# solutionsPath = os.path.join(cwd, "solutions/")
# defaultSolutionExtension = ".sol"
# sols = glob(solutionsPath + "*" + defaultSolutionExtension)
# solutions = []
# for sol in sols:
#     solutions.append(load_solution(sol))


problem = SudokuGeneticProblem(filename_initial_dict["s02b"])
(solution, number_of_generations, time_spent) = genetic_algorithm(problem)
print("NUMBER OF GENERATIONS: " + str(number_of_generations) + "\n" + "TIME SPENT: " + str(time_spent) + " seconds\n")