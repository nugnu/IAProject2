import sys
import time 
from collections import deque

from utils import *
from genetic_problems import *

# ______________________________________________________________________________
# Genetic Algorithm

def genetic_algorithm(problem, verbose=True):
    start = time.time()
    MIN_GEN_ITERATIONS = 175
    MAX_STALE_FACTOR = 12
    MIN_STALE = 20
    stale = 0
    problem.init_population()
    while True:
        prev_fitness_individual = max(problem.population, key=problem.fitness)
        if (problem.goal_test(prev_fitness_individual)): 
            if (verbose == True): print("GEN " + str(problem.gen) + " SOLUTION FOUND: \n" + problem.to_string(prev_fitness_individual) + "\n", "FITNESS = " + str(problem.fitness(prev_fitness_individual)) + "\n")
            return (prev_fitness_individual, problem.gen, time.time() - start)
        # print the best current individual
        if (verbose == True): print("GEN "    + str(problem.gen) + " BEST INDIVIDUAL: \n" + problem.to_string(prev_fitness_individual) + "\n", "FITNESS = " + str(problem.fitness(prev_fitness_individual)) + "\n")
        problem.replace_population(prev_fitness_individual) # replace the population and add the fittest of the previous generation to the new one 
        current_fitness_individual = max(problem.population, key = problem.fitness)
        
        prev_best_fitness = problem.fitness(prev_fitness_individual)
        current_best_fitness = problem.fitness(current_fitness_individual)
        
        KILL_PERCENT = 1
        MAX_STALE = 100 - (problem.MAX_FITNESS - current_best_fitness) * MAX_STALE_FACTOR
        if MAX_STALE < MIN_STALE: MAX_STALE = MIN_STALE

        if (current_best_fitness == prev_best_fitness):
            stale += 1
        else: stale = 0

        if (stale > MAX_STALE): 
            print("RE-SEEDING POPULATION\n")
            problem.stale_handler(KILL_PERCENT)
            stale = 0
        problem.gen += 1
