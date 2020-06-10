from collections import deque
from collections import Counter
from utils import *

class GeneticProblem:
   
    def __init__(self, gene_pool=None, initial=None, f_threshold=None, initial_population_size=30, 
    max_ngen=1000, mut_rate=0.2, crossover_rate=1, mut_type="swap", crossover_type="one_point"):
        self.gene_pool = gene_pool
        self.initial = initial
        self.f_threshold = f_threshold
        self.initial_population_size = initial_population_size
        self.max_ngen = max_ngen
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.mut_type = mut_type
        self.crossover_type = crossover_type
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

    def __init__(self, gene_pool=None, initial=None, f_threshold=None, initial_population_size=1000,
    max_ngen=1000, mut_rate=0.2, crossover_rate=1, mut_type="swap", crossover_type="one_point"):
        super().__init__(self, gene_pool, initial, f_threshold, initial_population_size, max_ngen, mut_rate,
        crossover_rate, mut_type, crossover_type)
        self.__SUDOKU_ARRAY_SIZE = 81
        self.__MAX_FITNESS = 1000000 # best possible fitness value -> to be fixed 
        self.gen = 0 
        self.population = []

    def fitness(self, individual): # TODO
        return 1
    
    def init_population(self):
        g = len(self.gene_pool)
        for i in range(self.pop_number):
            new_individual = [self.gene_pool[random.randrange(0, g)] for j in range(self.__SUDOKU_ARRAY_SIZE)]
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
        fitnesses = map(self.fitness, self.population)
        sampler = weighted_sampler(self.population, fitnesses)
        return [sampler() for i in range(r)]

    def crossover(self, x, y): 
        if random.uniform(0, 1) >= self.crossover_rate:
            return x # returns a individual from the current generation without doing the crossover 

        if (self.crossover_type == "one_point"):
            n = len(x)
            c = random.randrange(0, n)
            return x[:c] + y[c:]

        # TODO -> MORE crossover types 
        
        return x

    def mutate(self, x):
        if random.uniform(0, 1) >= self.mut_rate:
            return x

        if (self.mut_type == "swap"):
            n = len(x)
            g = len(self.gene_pool)
            c = random.randrange(0, n)
            r = random.randrange(0, g)
            new_gene = self.gene_pool[r]
            return x[:c] + [new_gene] + x[c + 1:]

        # TODO -> MORE mutation types 

        return x

    def replace_population(self, prev_fitness_individual=None):
        # apply crossover to 2 random individuals (more chance to the fittest) and them apply the mutation
        # do it len(population) times 
        self.population = [ mutate(crossover(*selection(2))) for i in range (len(self.population)) ] 
    
        # if needed, add the fittest of previous generation
        if (prev_fitness_individual != None):
            self.population.append(prev_fitness_individual)

    