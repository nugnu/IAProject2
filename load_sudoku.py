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

def load_solution(path):
    SUDOKU_SIZE = 81
    MAX_LINES = 11
    MAX_VALID_COLS = 9
    maze = open(path, 'r')
    ret = []
    
    counter_line = 0
    for line in maze:
        counter_col = 0
        counter_line = counter_line + 1
        if (counter_line > MAX_LINES): break
        for x in line:
            if x.isnumeric():
                counter_col = counter_col + 1
                if (counter_col > MAX_VALID_COLS): break
                number = int(x)
                if (number > 0 and number <= 9):
                    ret.append(number)
                else: assert 0 
    
    assert(len(ret) == SUDOKU_SIZE)
    ret = np.array(ret) # initial genetic problem attribute is expected to be a np array 
    return ret
