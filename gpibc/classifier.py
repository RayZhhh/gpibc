import copy
import random

import numpy as np
from numba import cuda

from gpibc.program import Program
from gpibc.fset import *
from .eval_gpu import GPUPopulationEvaluator
from .eval_cpu import CPUEvaluator


class BinaryClassifier:
    def __init__(self, train_set: np.ndarray, train_label: np.ndarray, test_set=None, test_label=None,
                 population_size=500, init_method='ramped_half_and_half', init_depth=(3, 6), max_program_depth=10,
                 generations=50, elist_size=5, tournament_size=5, crossover_prob=0.6, mutation_prob=0.3,
                 device='cuda:0', eval_batch=100):
        """
        Args:
            train_set        : dataset for training
            train_label      : label set for training
            test_set         : dataset for testing
            test_label       : label for testing
            population_size  : population size
            init_method      : 'full', 'growth', of 'ramped_half_and_half'
            init_depth       : the initial depth of the program
            max_program_depth: max program depth in the population, perform hoist mutation if program exceeds this
            generations      : evolution times
            elist_size       : the first 'elist_size' good individuals go directly to the next generation
            tournament_size  : tournament size while performing tournament selection
            crossover_prob   : crossover probability
            mutation_prob    : mutation probability
            device           : the cuda_device on which executes fitness evaluation
            eval_batch       : the number of program to evaluate simultaneously, valid when eval_method='population'
        """
        self.train_set = train_set
        self.train_label = train_label

        if len(train_set) != len(train_label):
            raise RuntimeError('The length of train set is not equal to the length of the train label set.')

        self.test_set = test_set
        self.test_label = test_label

        if test_set is not None and test_label is not None and len(test_set) != len(test_label):
            raise RuntimeError('The length of test set is not equal to the length of the test label set.')

        self.data_size = len(train_set)
        self.img_h = len(train_set[0])
        self.img_w = len(train_set[0][0])
        self.population_size = population_size
        self.init_method = init_method
        self.init_depth = init_depth
        self.max_program_depth = max_program_depth
        self.generations = generations
        self.elist_size = elist_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.device = device.split(':')[0]

        if self.device == 'cuda':
            self.device_id = int(device.split(':')[1])

        self.eval_batch = eval_batch

        if self.device == 'cuda' and not cuda.is_available():
            raise RuntimeError('Do not support CUDA on your cuda_device.')

        if self.device == 'cuda':
            self.evaluator = GPUPopulationEvaluator(self.train_set, self.train_label, self.eval_batch)
            cuda.select_device(self.device_id)

            # evaluator for the test set
            if self.test_set is not None:
                self.test_evaluator = GPUPopulationEvaluator(self.test_set, self.test_label)

        elif self.device == 'cpu':
            self.evaluator = CPUEvaluator(self.train_set, self.train_label)

        else:
            raise RuntimeError('Do not support such device.')

        # population properties
        self.population: List[Program] = []
        self.best_program: Program = ...
        self.best_program_in_each_gen: List[Program] = []
        self.best_fitness: float = ...
        #
        self.best_test_program: Program = ...

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

    def _update_generation_properties(self):
        self.best_program = max(self.population, key=lambda program: program.fitness)
        self.best_fitness = self.best_program.fitness
        self.best_program_in_each_gen.append(self.best_program)

    def _print_population_properties(self, gen):
        print('[ Generation   ] ', gen)
        print('[ Best Fitness ] ', self.best_fitness)
        print('[ Best Program ] ', self.best_program)
        print('')

    def train(self):
        self.best_program_in_each_gen = []

        # population initialization
        self.population_init()

        # evaluate fitness for the initial population
        self.evaluator.evaluate_population(self.population)

        # update
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
            self.evaluator.evaluate_population(self.population)

            # update
            self._update_generation_properties()
            self._print_population_properties(gen=iter_times)

    def run_test(self):
        self.test_evaluator.evaluate_population(self.best_program_in_each_gen)
        self.best_test_program = max(self.best_program_in_each_gen, key=lambda program: program.fitness)
        print(f'[ =========Run Test======== ]')
        print(f'[ Best program in test data ] {self.best_test_program}')
        print(f'[ Accuracy                  ] {self.best_test_program.fitness}')
