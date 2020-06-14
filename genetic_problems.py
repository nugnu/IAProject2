from collections import deque
from collections import Counter
from utils import *
import numpy as np
import copy 
import nakedSingles as nks

class GeneticProblem:
   
    def __init__(self, initial=None, f_threshold=None, initial_population_size=1, 
    max_ngen=10000, mut_rate=0.4, crossover_rate=1, mut_type="all", crossover_type="all",
    selection_type = "tournament", replacement_type = "all", init_type = "all"):
        self.initial = initial
        self.f_threshold = f_threshold
        self.initial_population_size = initial_population_size
        self.max_ngen = max_ngen
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.mut_type = mut_type
        self.crossover_type = crossover_type
        self.selection_type =  selection_type
        self.replacement_type = replacement_type
        self.init_type = init_type
        self.population = []

    def fitness(self):
        raise NotImplementedError

    def init_population(self):
        raise NotImplementedError

    def goal_test(self):
        raise NotImplementedError
    
    def selection(self):
        raise NotImplementedError

    def crossover(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError 

    def replace_population(self):
        raise NotImplementedError
    
class SudokuGeneticProblem(GeneticProblem):
   
    # INIT 

    def __get_init_dict(self):
        return {
            "random": self.__random_init,
            0: self.__random_init,
            "permutation": self.__permutation_init,
            1: self.__permutation_init,
            "row_permutation": self.__row_permutation_init,
            2: self.__row_permutation_init
        }

    def __get_crossover_dict(self):
        return {
            0: self.__one_point_crossover,
            "one_point": self.__one_point_crossover,
            1: self.__vertical_two_point_crossover,
            "vertical_two_point": self.__vertical_two_point_crossover,
            2: self.__vertical_one_point_crossover,
            "vertical_one_point":   self.__vertical_one_point_crossover,
            3: self.__vertical_one_point_crossover
        }

    def __get_selection_dict(self):
        return {
            0: self.__tournament,
            "tournament": self.__tournament,
            1: self.__roulette,
            "roulette": self.__roulette
        }

    def __get_replacement_dict(self):
        return {
            "default": self.__default_replace,
            0: self.__default_replace,
            "elitism": self.__elitism_replace,
            1: self.__elitism_replace
        }

    def __get_mutation_dict(self):
        return {
            "flip": self.__flip_mutate,
            0: self.__flip_mutate,
            "nflip": self.__nflip_mutate,
            1: self.__nflip_mutate,
            "swap": self.__swap_mutate,
            2: self.__swap_mutate,
            "nswap": self.__nswap_mutate,
            3: self.__nswap_mutate,
            "horizontal_swap": self.__horizontal_swap_mutate,
            4: self.__horizontal_swap_mutate,
            "horizontal_nswap": self.__horizontal_nswap_mutate,
            5: self.__horizontal_nswap_mutate
        }
    
    def __pre_processing(self):
        self.initial.shape = (9,9)
        self.initial = nks.nakedSingles(self.initial.tolist())
        self.initial = np.array(self.initial)
        self.initial.shape = (81)
        return self.initial

    def __initEmptyArray(self):
        return np.sort(np.array(np.nonzero(self.initial == 0)[0])) # returns the indices of the zero elements on the sudoku puzzle

    def __isRowPreserved(self):
        if ((self.mut_type == "horizontal_swap" or self.mut_type == "horizontal_nswap") and 
        (self.crossover_type == "vertical_two_point" or self.crossover_type == "vertical_one_point") and
        (self.init_type == "row_permutation")) :
            return True
        return False

    def __init__(self, initial, f_threshold=None, initial_population_size=1000,
    max_ngen=10000, mut_rate=0.5, crossover_rate=1, mut_type="horizontal_swap", crossover_type="vertical_one_point",
    selection_type = "tournament", replacement_type = "elitism", init_type = "row_permutation"):
        super().__init__(initial, f_threshold, initial_population_size, max_ngen, mut_rate,
        crossover_rate, mut_type, crossover_type, selection_type, replacement_type, init_type)
        self.__SUDOKU_ARRAY_SIZE = 81
        self.__MAX_SWAP = 3 # max possible swaps to be done on a single mutation
        self.__MAX_FLIP = 3  # max possible random gene change to be done on a single mutation
        self.init_dict = self.__get_init_dict()
        self.selection_dict = self.__get_selection_dict()
        self.crossover_dict = self.__get_crossover_dict()
        self.replacement_dict = self.__get_replacement_dict()
        self.mutation_dict = self.__get_mutation_dict()
        self.MAX_FITNESS = 243 # best possible fitness value 
        self.gene_pool = range(1,10)
        self.gen = 0 
        self.population = []
        print("INITIAL PROBLEM:\n" + self.to_string(self.initial) + "\n")
        self.initial = self.__pre_processing()
        print("INITIAL PROBLEM AFTER NAKED_SINGLES PRE-PROCESSING:\n" + self.to_string(self.initial) + "\n")
        self.emptyArray = self.__initEmptyArray() # each position of this array keeps the position of an empty position on a sudoku puzzle 
        self.row_preserved = self.__isRowPreserved()

    # GENERAL   

    def __getRow(self, individual, i):
        return individual[i:i+9]

    def __getRows(self, individual):
        return [ self.__getRow(individual, i) for i in (np.arange(0,9) *9) ]

    def __getCol(self, individual, i):
        return individual[i:(72+i+1):9]

    def __getCols(self, individual):
        return [self.__getCol(individual,i) for i in range(9)]

    def __getBox(self, individual, i):
        return list(individual[i:i+3]) + list(individual[i+9:i+12]) + list(individual[i+18:i+21])

    def __getBoxes(self, individual):
        return [self.__getBox(individual, i) for i in [0, 3, 6, 27, 30, 33, 54, 57, 60]]

    # GOAL TEST 

    def goal_test(self, individual):
        if (self.f_threshold != None):
            if (self.max_ngen != None):
                return (self.gen == self.max_ngen or self.fitness(individual) >= self.f_threshold)
            else:
                return (self.fitness(individual) >= self.f_threshold)
        else: # no threshold -> its expecting an actual solution (best possible fitness)
            if (self.max_ngen != None):
                return (self.gen == self.max_ngen or self.fitness(individual) == self.MAX_FITNESS)
            else:
                return (self.fitness(individual) == self.MAX_FITNESS)
    
    # FITNESS
   
    def fitness(self, individual):
        if (self.row_preserved == True):
            return (
                sum([len(set(col)) for col in self.__getCols(individual)]) +
                sum([len(set(box)) for box in self.__getBoxes(individual)]) +
                81
            )
        return ( 
            sum([len(set(row)) for row in self.__getRows(individual)]) +
            sum([len(set(col)) for col in self.__getCols(individual)]) +
            sum([len(set(box)) for box in self.__getBoxes(individual)]) 
        )

    # INIT POPULATION

    def __permutation_init(self):
        # every solution is a permutation of (1,2,3,4,5,6,7,8,9,1,2,3...)
        permutation = []
        for _ in range(9):
            permutation += np.arange(1,10).tolist()
        # as we only want to solve the sudoku using the empty (0) spaces, we remove ocurrences of
        # non zero initial values from the permutation
        for number in self.initial[self.initial != 0]:
            permutation.remove(number)
        # now for each individual, shuffle this permutation and fill the empty spaces with it
        for i in range(self.initial_population_size):
            random.shuffle(permutation)
            permut_iterator = 0
            new_individual = []
            for j in range(self.__SUDOKU_ARRAY_SIZE):
                if (j not in self.emptyArray):
                    new_individual.append(self.initial[j])  
                else:
                    new_individual.append(permutation[permut_iterator])
                    permut_iterator += 1
            self.population.append(new_individual)

    def __row_permutation_init(self): 
        # it has a similar idea of __permutation_init(). But rows are guaranteed to have no duplicates 
        for _ in range(self.initial_population_size):
            new_individual = copy.deepcopy(self.initial)
            row_iterator = 0
            np_emptyArray = np.array(self.emptyArray)
            for row in self.__getRows(self.initial):
                permut_iterator = 0 
                row_permutation = np.arange(1,10).tolist()
                for number in row[row != 0]:
                    row_permutation.remove(number)
                random.shuffle(row_permutation)
                for empty_pos in np_emptyArray[(np_emptyArray >= row_iterator) & (np_emptyArray <= row_iterator + 8)]:
                    new_individual[empty_pos] = row_permutation[permut_iterator]
                    permut_iterator += 1
                row_iterator += 9
            self.population.append(new_individual.tolist())

    def __random_init(self):
        g = len(self.gene_pool)
        for i in range(self.initial_population_size):
            new_individual = []
            for j in range(self.__SUDOKU_ARRAY_SIZE):
                if (j not in self.emptyArray):
                    new_individual.append(self.initial[j])  
                else:
                    new_individual.append(self.gene_pool[random.randrange(0, g)]) 
            self.population.append(new_individual)
    
    def init_population(self):
        m = 3 # number of methods

        if (self.init_type == "all"):
            c = random.randrange(0, m)
            self.init_dict[c]()

        return self.init_dict[self.init_type]()
            
    # SELECTION
    
    def __tournament(self, r):
        k = 2
        return [self.__tournament_util(k) for i in range(r)]

    def __tournament_util(self, k):
        competitors = []
        for _ in range(k):            
            pop_len = len(self.population)
            c = random.randrange(pop_len)
            competitors.append(self.population[c])
        return max(competitors, key=self.fitness)

    def __roulette(self, r):
        fitnesses = map(self.fitness, self.population)
        sampler = weighted_sampler(self.population, fitnesses)
        return [sampler() for i in range(r)]

    def selection(self, r):
        m = 2 # number of methods

        if (self.selection_type == "all"):
            c = random.randrange(0, m)
            return self.selection_dict[c](r)
        
        return self.selection_dict[self.selection_type](r)

    # RECOMBINATION

    def __one_point_crossover(self, x, y):
        c = random.randrange(0, len(self.emptyArray))
        pos = self.emptyArray[c]
        return x[:pos] + y[pos:]

    # vertical crossovers preserve rows

    def __vertical_one_point_crossover(self, x, y):
        crossover_point = random.randint(0,9) * 9
        return x[:crossover_point] + y[crossover_point:]

    def __vertical_two_point_crossover(self, x, y):
        crossover_point1 = random.randint(0, 8)
        crossover_point2 = random.randint(crossover_point1 + 1, 9)
        crossover_point1 = crossover_point1*9
        crossover_point2 = crossover_point2*9
        return x[:crossover_point1] + y[crossover_point1:crossover_point2] + x[crossover_point2:]

    def crossover(self, x, y): 
        m = 3 # number of methods

        if random.uniform(0, 1) >= self.crossover_rate or self.crossover_type == None:
            return x # returns a individual from the current generation without doing the crossover 

        if (self.crossover_type == "all"):
            c = random.randrange(0, m)
            return self.crossover_dict[c](x, y)

        return self.crossover_dict[self.crossover_type](x, y)
                    
    # MUTATION

    def __flip_mutate(self, x):
        g = len(self.gene_pool)
        c = random.randrange(0, len(self.emptyArray)) # select a random position of the empty array
        new_gene = self.gene_pool[random.randrange(0,g)]
        x[self.emptyArray[c]] = new_gene
        return x
    
    def __nflip_mutate(self, x):
        n = random.randrange(1, self.__MAX_FLIP)
        for _ in range(n):
            g = len(self.gene_pool)
            c = random.randrange(0, len(self.emptyArray)) # select a random position of the empty array
            new_gene = self.gene_pool[random.randrange(0,g)]
            x[self.emptyArray[c]] = new_gene
        return x

    def __swap_mutate(self, x):
        empty_array_len = len(self.emptyArray)
        c1 = random.randrange(0, empty_array_len) # select a random position of the empty array
        c2 = random.randrange(0, empty_array_len) # select another random position of the empty array
        tmp = x[self.emptyArray[c1]]
        x[self.emptyArray[c1]] = x[self.emptyArray[c2]]
        x[self.emptyArray[c2]] = tmp
        return x

    def __nswap_mutate(self, x):
        empty_array_len = len(self.emptyArray)
        n = random.randrange(1, self.__MAX_SWAP)
        for _ in range(n):
            c1 = random.randrange(0, empty_array_len) # select a random position of the empty array
            c2 = random.randrange(0, empty_array_len) # select another random position of the empty array
            tmp = x[self.emptyArray[c1]]
            x[self.emptyArray[c1]] = x[self.emptyArray[c2]]
            x[self.emptyArray[c2]] = tmp
        return x

    def __horizontal_swap_mutate(self, x):
        # choose randomly which row its gonna pick to perform the swap
        row = random.randrange(0,9)
        start_pos = np.searchsorted(self.emptyArray, 9*row) # the position on the empty array that the row starts
        end_pos = np.searchsorted(self.emptyArray, 9*row + 9) # the position on the empty array that the row ends
        c1 = random.randrange(start_pos, end_pos)
        c2 = random.randrange(start_pos, end_pos)
        tmp = x[self.emptyArray[c1]]
        x[self.emptyArray[c1]] = x[self.emptyArray[c2]]
        x[self.emptyArray[c2]] = tmp
        return x

    def __horizontal_nswap_mutate(self, x):
        # choose randomly which row its gonna pick to perform the swap
        n = random.randrange(1, self.__MAX_SWAP)
        for _ in range(n):
            row = random.randrange(0,9)
            start_pos = np.searchsorted(self.emptyArray, 9*row) # the position on the empty array that the row starts
            end_pos = np.searchsorted(self.emptyArray, 9*row + 9) # the position on the empty array that the row ends
            c1 = random.randrange(start_pos, end_pos)
            c2 = random.randrange(start_pos, end_pos)
            tmp = x[self.emptyArray[c1]]
            x[self.emptyArray[c1]] = x[self.emptyArray[c2]]
            x[self.emptyArray[c2]] = tmp
        return x

    def mutate(self, x):
        m = 5 # number of methods

        if self.goal_test(x) == True: # dont mutate
            return x

        if  random.uniform(0, 1) >= self.mut_rate or self.mut_type == None:
            return x

        if  (self.mut_type == "all"):
            c = random.randrange(0, m)
            return self.mutation_dict[c](x)
        
        return self.mutation_dict[self.mut_type](x)

    # REPLACE POPULATION

    def __default_replace(self, prev_fitness_individual):
        # apply crossover to 2 random individuals (more chance to the fittest) and them apply the mutation
        # do it len(population) times 
        self.population = [ self.mutate(self.crossover(*self.selection(2))) for i in range (len(self.population)) ] 
    
        # if needed, add the fittest of previous generation
        if (prev_fitness_individual != None):
            self.population.append(prev_fitness_individual)

    def __elitism_replace(self, prev_fitness_individual):  # similar to default but the population doesnt get bigger (same size as initial population)
        # apply crossover to 2 random individuals (more chance to the fittest) and them apply the mutation
        # do it len(population) times 
        self.population = [ self.mutate(self.crossover(*self.selection(2))) for i in range (len(self.population) - 1 ) ] 
    
        # if needed, add the fittest of previous generation
        if (prev_fitness_individual != None):
            self.population.append(prev_fitness_individual)


    def replace_population(self, prev_fitness_individual=None):
        m = 2 # number of methods

        if  (self.replacement_type == "all"):
            c = random.randrange(0, m)
            return self.replacement_dict[c](prev_fitness_individual)
        
        return self.replacement_dict[self.replacement_type](prev_fitness_individual)
           
    # STALE

    def kill_population(self, percent):
        n = int(percent * len(self.population))
        self.population = self.population[n:]

    def stale_handler(self, percent):
        self.kill_population(percent)
        self.init_population()
  
    # DISPLAY

    def to_string(self, individual):
        np_individual = np.array(individual)
        np_individual.shape = (9,9)
        return str(np_individual)
    