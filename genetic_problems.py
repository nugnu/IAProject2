from collections import deque
from collections import Counter
from utils import *
import numpy as np

class GeneticProblem:
   
    def __init__(self, initial=None, f_threshold=None, initial_population_size=30, 
    max_ngen=1000, mut_rate=0.2, crossover_rate=1, mut_type="swap", crossover_type="one_point",
    selection_type = "tournament", replacement_type = "default"):
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
    
class SudokuGeneticProblem(GeneticProblem):

    def __init__(self, initial, f_threshold=None, initial_population_size=1000,
    max_ngen=1000, mut_rate=0.2, crossover_rate=1, mut_type="swap", crossover_type="one_point",
    selection_type = "tournament", replacement_type = "default"):
        super().__init__(self, initial, f_threshold, initial_population_size, max_ngen, mut_rate,
        crossover_rate, mut_type, crossover_type, selection_type, replacement_type)
        self.__SUDOKU_ARRAY_SIZE = 81
        self.__MAX_FITNESS = 1000000 # best possible fitness value -> to be fixed 
        self.__MAX_SWAP = 10 # max possible swaps to be done on a single mutation
        self.__MAX_FLIP = 10 # max possible random gene change to be done on a single mutation
        self.gene_pool = range(1,10)
        self.gen = 0 
        self.population = []
        self.emptyArray = __initEmptyArray() # each position of this array keeps the position of an empty position on a sudoku puzzle 

    def __initEmptyArray(self):
        return np.nonzero(self.initial == 0) # returns the indices of the zero elements on the sudoku puzzle

    def __tournament(self, k):
        competitors = []
        for _ in range(k):            
            pop_len = len(self.population)
            c = random.randrange(pop_len)
            competitors.append(population[c])
        return max(competitors, key=self.fitness)

    def fitness(self, individual): # TODO
        return 1
    
    def init_population(self):
        g = len(self.gene_pool)
        for i in range(self.pop_number):
            new_individual = []
            for j in range(self.__SUDOKU_ARRAY_SIZE):
                if (j not in self.emptyArray):
                    new_individual[j] = self.initial[j]
                else:
                    new_individual[j] = self.gene_pool[random.randrange(0, g)]
            self.population.append(new_individual)
            
    def goal_test(self, individual):
        if (self.f_threshold != None):
            if (self.max_ngen != None):
                return (self.gen == self.max_ngen or fitness(individual) >= self.f_threshold)
            else:
                return (fitness(individual) >= self.f_threshold)
        else: # no threshold -> its expecting an actual solution (best possible fitness)
            if (self.max_ngen != None):
                return (self.gen == self.max_ngen or fitness(individual) == self.__MAX_FITNESS)
            else:
                return (fitness(individual) == self.__MAX_FITNESS)

    def selection(self, r):
        if (self.selection_type == "roulette"):
            fitnesses = map(self.fitness, self.population)
            sampler = weighted_sampler(self.population, fitnesses)
            return [sampler() for i in range(r)]
        
        if (self.selection_type == "tournament"):
            k = 3 # number of tournament competitors
            return [self.__tournament(k) for i in range(r)]

    def crossover(self, x, y): 
        if random.uniform(0, 1) >= self.crossover_rate:
            return x # returns a individual from the current generation without doing the crossover 

        if (self.crossover_type == "one_point"):
            c = random.randrange(0, len(self.emptyArray))
            pos = self.emptyArray[c]
            return x[:pos] + y[pos:]

        # TODO -> MORE crossover types 
        
        return x

    def mutate(self, x):
        if random.uniform(0, 1) >= self.mut_rate:
            return x

        if (self.mut_type == "flip"): # changes a random gene from the individual
            g = len(self.gene_pool)
            c = random.randrange(0, len(self.emptyArray)) # select a random position of the empty array
            new_gene = self.gene_pool[random.randrange(0,g)]
            x[self.emptyArray[c]] = new_gene
            return x

        if (self.mut_type == "nflip"): # perform n (1 up to __MAX_FLIP) flips    
            n = random.randrange(1, self.__MAX_FLIP)
            for _ in range(n):
                g = len(self.gene_pool)
                c = random.randrange(0, len(self.emptyArray)) # select a random position of the empty array
                new_gene = self.gene_pool[random.randrange(0,g)]
                x[self.emptyArray[c]] = new_gene
            return x

        if (self.mut_type == "swap"): # swap two genes from the individual
            empty_array_len = len(self.emptyArray)
            c1 = random.randrange(0, empty_array_len) # select a random position of the empty array
            c2 = random.randrange(0, empty_array_len) # select another random position of the empty array
            tmp = x[self.emptyArray[c1]]
            x[self.emptyArray[c1]] = x[self.emptyArray[c2]]
            x[self.emptyArray[c2]] = tmp
            return x

        if (self.mut_type == "nswap"): # performs swap n (1 up to __MAX_SWAP) times 
            empty_array_len = len(self.emptyArray)
            n = random.randrange(1, self.__MAX_SWAP)
            for _ in range(n):
                c1 = random.randrange(0, empty_array_len) # select a random position of the empty array
                c2 = random.randrange(0, empty_array_len) # select another random position of the empty array
                tmp = x[self.emptyArray[c1]]
                x[self.emptyArray[c1]] = x[self.emptyArray[c2]]
                x[self.emptyArray[c2]] = tmp
            return x

        # TODO -> MORE mutation types 

        return x

    def replace_population(self, prev_fitness_individual=None):
        if (self.replacement_type == "default"):
            # apply crossover to 2 random individuals (more chance to the fittest) and them apply the mutation
            # do it len(population) times 
            self.population = [ mutate(crossover(*selection(2))) for i in range (len(self.population)) ] 
        
            # if needed, add the fittest of previous generation
            if (prev_fitness_individual != None):
                self.population.append(prev_fitness_individual)

        if (self.replacement_type == "elitism"): # similar to default but the population doesnt get bigger (same size as initial population)
            # apply crossover to 2 random individuals (more chance to the fittest) and them apply the mutation
            # do it len(population) times 
            self.population = [ mutate(crossover(*selection(2))) for i in range (len(self.population) - 1 ) ] 
        
            # if needed, add the fittest of previous generation
            if (prev_fitness_individual != None):
                self.population.append(prev_fitness_individual)

    