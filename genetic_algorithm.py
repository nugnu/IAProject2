import sys
from collections import deque

from utils import *
from genetic_problems import *

# ______________________________________________________________________________
# Genetic Algorithm

def genetic_algorithm(problem):
    DELTA = 5 # measure how far a population is far from the solution
    MIN_GEN_ITERATIONS = 175
    problem.init_population()
    stale = 0
    current_gen_iteration = 0
    while True:
        prev_fitness_individual = max(problem.population, key=problem.fitness)
        if (problem.goal_test(prev_fitness_individual)): 
            print("GEN " + str(problem.gen) + " SOLUTION FOUND: \n" + problem.print_individual(prev_fitness_individual) + "\n", "FITNESS = " + str(problem.fitness(prev_fitness_individual)) + "\n")
            return prev_fitness_individual
        # print the best current individual
        print("GEN "    + str(problem.gen) + " BEST INDIVIDUAL: \n" + problem.print_individual(prev_fitness_individual) + "\n", "FITNESS = " + str(problem.fitness(prev_fitness_individual)) + "\n")
        problem.replace_population(prev_fitness_individual) # replace the population and add the fittest of the previous generation to the new one 
        current_fitness_individual = max(problem.population, key = problem.fitness)
        
        prev_best_fitness = problem.fitness(prev_fitness_individual)
        current_best_fitness = problem.fitness(current_fitness_individual)
        
        KILL_PERCENT = 1
        MAX_STALE = 150
        if (current_best_fitness == prev_best_fitness):
            stale += 1
            if (current_best_fitness < problem.MAX_FITNESS - DELTA and current_gen_iteration > MIN_GEN_ITERATIONS):
                # bad population
                MAX_STALE = 20
        else: stale = 0

        if (stale > MAX_STALE): 
            print("RE-SEEDING POPULATION\n")
            problem.stale_handler(KILL_PERCENT)
            stale = 0
            current_gen_iteration = 0
        problem.gen += 1
        current_gen_iteration += 1
