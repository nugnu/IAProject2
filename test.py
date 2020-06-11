from genetic_problems import *
from genetic_algorithm import *
import numpy as np
from load_puzzle import *
import os
from glob import glob 

cwd = os.getcwd()
puzzlesPath = os.path.join(cwd, "puzzles/")
defaultPuzzleExtension = ".in"
puzzles = glob(puzzlesPath + "*" + defaultPuzzleExtension)
initials = []

for puzzle in puzzles:
    initials.append(load_puzzle(puzzle))

print(len(initials))
for initial in initials:
    print(initial)
    print(len(initial))