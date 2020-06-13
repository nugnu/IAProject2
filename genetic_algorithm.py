import sys
from collections import deque

from utils import *
from genetic_problems import *

# ______________________________________________________________________________
# Genetic Algorithm

def genetic_algorithm(problem):
    MAX_STALE = 100
    DELTA = 5 # measure how far a population is far from the solution
    problem.init_population()
    stale = 0
    while True:
        prev_fitness_individual = max(problem.population, key=problem.fitness)
        if (problem.goal_test(prev_fitness_individual)): 
            print("GEN " + str(problem.gen) + " SOLUTION FOUND: \n" + problem.print_individual(prev_fitness_individual) + "\n", "FITNESS = " + str(problem.fitness(prev_fitness_individual)) + "\n")
            return prev_fitness_individual
        # print the best current individual
        print("GEN "    + str(problem.gen) + " BEST INDIVIDUAL: \n" + problem.print_individual(prev_fitness_individual) + "\n", "FITNESS = " + str(problem.fitness(prev_fitness_individual)) + "\n")
        problem.replace_population(prev_fitness_individual) # replace the population and add the fittest of the previous generation to the new one 
        current_fitness_individual = max(problem.population, key = problem.fitness)
        
        if (problem.fitness(prev_fitness_individual) == problem.fitness(current_fitness_individual)):
            stale += 1
        else: stale = 0

        if (stale > MAX_STALE): 
            KILL_PERCENT = 1
            if (problem.fitness(current_fitness_individual) < problem.MAX_FITNESS - DELTA):
                KILL_PERCENT = 1
            print("RE-SEEDING POPULATION\n")
            problem.stale_handler(KILL_PERCENT)
            stale = 0
        problem.gen += 1
