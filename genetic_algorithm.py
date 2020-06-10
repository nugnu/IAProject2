import sys
from collections import deque

from utils import *
from genetic_problems import *

# ______________________________________________________________________________
# Genetic Algorithm

def genetic_algorithm(problem):
    problem.init_population()
    while True:
        prev_fitness_individual = max(problem.population, key=problem.fitness)
        if (problem.goal_test(prev_fitness_individual)): 
            print("GEN " + str(problem.gen) + " SOLUTION FOUND: \n" + str(prev_fitness_individual) + "\n", "FITNESS = " + str(problem.fitness(prev_fitness_individual)) + "\n")
            return prev_fitness_individual
        # print the best current individual
        print("GEN " + str(problem.gen) + " BEST INDIVIDUAL: \n" + str(prev_fitness_individual) + "\n", "FITNESS = " + str(problem.fitness(prev_fitness_individual)) + "\n")

        problem.replace_population(prev_fitness_individual) # replace the population and add the fittest of the previous generation to the new one 
        problem.gen = problem.gen + 1
