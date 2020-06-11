import numpy as np

def load_puzzle(path):
    SUDOKU_SIZE = 81
    maze = open(path, 'r')
    ret = []
    
    for line in maze:
        for x in line:
            if x.isnumeric():
                number = int(x)
                if (number >= 0 and number <= 9):
                    ret.append(number)
                else: assert 0 
    
    assert(len(ret) == SUDOKU_SIZE)
    ret = np.array(ret) # initial genetic problem attribute is expected to be a np array 
    return ret
