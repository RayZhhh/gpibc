import copy
import random
from typing import List

import numba.cuda
import numpy as np

from program import Program
from fset import *
from evaluator import GPUProgramEvaluator, GPUPopulationEvaluator


class BinaryClassifier:
    def __init__(self, dataset: np.ndarray, label: np.ndarray, population_size=500, init_method='ramped_half_and_half',
                 init_depth=(3, 6), max_program_depth=6, generations=50, elist_size=5, tournament_size=5,
                 crossover_prob=0.6, mutation_prob=0.3, device='cuda', eval_method='program'):
        """
        Args:
            dataset          : dataset
            label            : label set
            population_size  : population size
            init_method      : 'full', 'growth', of 'ramped_half_and_half'
            init_depth       : the initial depth of the program
            max_program_depth: max program depth in the population, perform hoist mutation if program exceeds this
            generations      : evolution times
            elist_size       : the first 'elist_size' good individuals go directly to the next generation
            tournament_size  : tournament size while performing tournament selection
            crossover_prob   : crossover probability
            mutation_prob    : mutation probability
            device           : the device on which executes fitness evaluation
            eval_method      : 'program' or 'population', device will evaluate a program or a population simultaneously
        """
        self.dataset = dataset
        self.label = label

        if len(dataset) != len(label):
            raise RuntimeError('The length of dataset is not equal to the length of the label set.')

        self.data_size = len(dataset)
        self.img_h = len(dataset[0])
        self.img_w = len(dataset[0][0])
        self.population_size = population_size
        self.init_method = init_method
        self.init_depth = init_depth
        self.max_program_depth = max_program_depth
        self.generations = generations
        self.elist_size = elist_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.device = device
        self.eval_method = eval_method

        if device == 'cuda' and not numba.cuda.is_available():
            raise RuntimeError('Do not support CUDA on your device.')

        if device == 'cuda':
            if self.eval_method == 'program':
                self.gpu_evaluator_program = GPUProgramEvaluator(self.dataset, self.label)
            elif self.eval_method == 'population':
                self.gpu_evaluator_population = GPUPopulationEvaluator(self.dataset, self.label)
            else:
                raise RuntimeError('Error: eval_method must be \'program\' or \'population\'.')

        # population properties
        self.population: List[Program] = []
        self.best_program: Program = ...
        self.best_fitness: float = ...

    def population_init(self):
        if self.init_method == 'full':
            for _ in range(self.population_size):
                rand_depth = random.randint(self.init_depth[0], self.init_depth[1])
                self.population.append(Program(self.img_h, self.img_w, rand_depth, 'full'))

        elif self.init_method == 'growth':
            for _ in range(self.population_size):
                rand_depth = random.randint(self.init_depth[0], self.init_depth[1])
                self.population.append(Program(self.img_h, self.img_w, rand_depth, 'growth'))
        else:  # 'ramped_half_and_half'
            full_num = int(self.population_size / 2)
            growth_num = self.population_size - full_num

            for _ in range(full_num):
                rand_depth = random.randint(self.init_depth[0], self.init_depth[1])
                self.population.append(Program(self.img_h, self.img_w, rand_depth, 'full'))

            for _ in range(growth_num):
                rand_depth = random.randint(self.init_depth[0], self.init_depth[1])
                self.population.append(Program(self.img_h, self.img_w, rand_depth, 'growth'))

    def tournament_selection(self) -> Program:
        plist = random.sample(population=self.population, k=self.tournament_size)
        selected_program = max(plist, key=lambda program: program.fitness)
        return copy.deepcopy(selected_program)

    def mutation(self, program) -> Program:
        # performing crossover or mutation according to the prob
        prob = random.random()
        if prob < self.crossover_prob:
            donor = self.tournament_selection()
            program.crossover(donor=donor)
        elif prob < self.crossover_prob + self.mutation_prob:
            if random.random() < 0.5:
                program.point_mutation()
            else:
                program.subtree_mutation()

        # perform hoist mutation multiple times
        # until its depth less equal than self.max_program_depth
        while program.depth > self.max_program_depth:
            program.hoist_mutation()

        return program

    def _fitness_evaluation_cpu(self, program: Program):
        pass

    def _fitness_evaluation_gpu(self, program: Program):
        self.gpu_evaluator_program.fitness_evaluate(program)

    def _fitness_evaluatrion_gpu_pop(self):
        self.gpu_evaluator_population.fitness_evaluate(self.population)

    def _update_generation_properties(self):
        self.best_program = self.population[0]
        self.best_fitness = self.population[0].fitness

        for i in range(self.population_size):
            if self.population[i].fitness > self.best_fitness:
                self.best_fitness = self.population[i].fitness
                self.best_program = self.population[i]

    def _print_population_properties(self, gen):
        print('[ Generation   ] ', gen)
        print('[ Best fitness ] ', self.best_fitness)
        print('[ Best program ] ', self.best_program)
        print('')

    def train(self):

        # population initialization
        self.population_init()

        # evaluate fitness for the initial population
        if self.device == 'cuda':
            if self.eval_method == 'program':
                for program in self.population:
                    self._fitness_evaluation_gpu(program)
            else:
                self._fitness_evaluatrion_gpu_pop()
        else:
            for program in self.population:
                self._fitness_evaluation_cpu(program)

        self._update_generation_properties()
        self._print_population_properties(gen=0)

        # do iterations
        for iter_times in range(1, self.generations):
            new_population: List[Program] = []

            # elitism
            temp_pop = copy.deepcopy(self.population)
            temp_pop.sort(key=lambda program: program.fitness, reverse=True)
            for i in range(self.elist_size):
                new_population.append(temp_pop[i])

            # generate new population
            for i in range(self.population_size - self.elist_size):
                # selection
                program = self.tournament_selection()
                # mutation
                new_program = self.mutation(program)
                new_population.append(new_program)

            # update properties of the population
            self.population = new_population

            # fitness evaluation
            if self.device == 'cuda':
                if self.eval_method == 'program':
                    for program in self.population:
                        self._fitness_evaluation_gpu(program)
                else:
                    self._fitness_evaluatrion_gpu_pop()
            else:
                for program in self.population:
                    self._fitness_evaluation_cpu(program)

            # update
            self._update_generation_properties()
            self._print_population_properties(gen=iter_times)
