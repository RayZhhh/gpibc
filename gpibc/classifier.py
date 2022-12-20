import copy
import random
import time
from typing import List

import numpy as np

from gpibc.program import Program
from .eval_cpu import CPUEvaluator


class BinaryClassifier:
    def __init__(self, train_set: np.ndarray, train_label: np.ndarray, test_set=None, test_label=None,
                 population_size=500, init_method='ramped_half_and_half', init_depth=(3, 6), max_program_depth=8,
                 generations=50, elist_size=5, tournament_size=7, crossover_prob=0.8, mutation_prob=0.19,
                 fit_criterion='accuracy', device='py_cuda', cu_arch=None, cu_code=None, eval_batch=10, thread_per_block=128):
        """
        Args:
            train_set        : dataset for training
            train_label      : label set for training
            test_set         : dataset for testing
            test_label       : label for testing
            population_size  : population size
            init_method      : 'full', 'grow', of 'ramped_half_and_half'
            init_depth       : the initial depth of the program
            max_program_depth: max program depth in the population, perform hoist mutation if program exceeds this
            generations      : evolution times
            elist_size       : the first 'elist_size' good individuals go directly to the next generation
            tournament_size  : tournament size while performing tournament selection
            crossover_prob   : crossover probability
            mutation_prob    : mutation probability
            fit_criterion    : the criterion to evaluate the program, such as accuracy or neg_bce loss
            device           : the cuda_device on which executes fitness evaluation
            cu_arch          : [effective if using py_cuda] cuda arch used in: nvcc -o -arch=compute_75
            cu_code          : [effective if using py_cuda] cuda code used in: nvcc -o -code=sm_75
            eval_batch       : the number of program to evaluate simultaneously, valid when eval_method='population'
            thread_per_block : blockDim.x
        """
        if len(train_set) != len(train_label):
            raise RuntimeError('The length of train set is not equal to the length of the train label set.')

        if test_set is not None and test_label is not None and len(test_set) != len(test_label):
            raise RuntimeError('The length of test set is not equal to the length of the test label set.')

        if init_method not in ['full', 'grow', 'ramped_half_and_half']:
            raise RuntimeError('Argument "init_method" must be "full" or "grow" or "ramped_half_and_half".')

        if fit_criterion not in ['accuracy', 'neg_bce']:
            raise RuntimeError('Argument "fit_criterion" must be "accuracy" or "neg_bce".')

        if device not in ['py_cuda', 'numba_cuda', 'cupy', 'cpu']:
            raise RuntimeError('Argument "device" must be "py_cuda" or "numba_cuda" or "cupy" or "cpu".')

        self.train_set = train_set
        self.train_label = train_label
        self.test_set = test_set
        self.test_label = test_label
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
        self.device = device
        self.cu_arch = cu_arch
        self.cu_code = cu_code
        self.eval_batch = eval_batch
        self.thread_per_block = thread_per_block
        self.cuda_kernel_time = 0
        self.fit_criterion = fit_criterion

        # program evaluator
        self.evaluator = self._get_evaluator(self.train_set, self.train_label, is_for_train=True)
        if self.test_set is not None:
            self.test_evaluator = self._get_evaluator(self.test_set, self.test_label, is_for_train=False)

        # population properties
        self.population: List[Program] = []
        self.best_program: Program = ...
        self.best_program_in_each_gen: List[Program] = []
        self.best_fitness: float = ...
        self.best_test_program: Program = ...

        # performance
        self.fitness_evaluation_time = 0
        self.training_time = 0

    def _get_evaluator(self, data, label, is_for_train=True):
        """Returns an evaluator for train_set or test_set.
        """
        eval_batch_ = self.eval_batch if is_for_train else 1

        # the evaluator for test calculate the accuracy
        if not is_for_train:
            metric_ = 'accuracy'
        else:
            metric_ = self.fit_criterion

        if self.device == 'numba_cuda':
            from .eval_numba_cuda import NumbaCudaEvaluator
            return NumbaCudaEvaluator(data, label, eval_batch_, self.thread_per_block, fit_criterion=metric_)

        elif self.device == 'cpu':
            return CPUEvaluator(data, label)

        elif self.device == 'py_cuda':
            from .eval_pycuda import PyCudaEvaluator
            return PyCudaEvaluator(data, label, eval_batch_, self.thread_per_block, metric_, self.cu_arch, self.cu_code)

        else:  # 'cupy'
            from .eval_cupy import CuPyEvaluator
            return CuPyEvaluator(data, label, eval_batch_, self.thread_per_block)

    def _shuffle_dataset_and_label(self):
        data_l = list(zip(self.train_set, self.train_label))
        np.random.shuffle(data_l)
        dataset_, label_ = zip(*data_l)
        self.train_set = dataset_
        self.train_label = label_

    def population_init(self):
        if self.init_method == 'full':
            for _ in range(self.population_size):
                rand_depth = random.randint(self.init_depth[0], self.init_depth[1])
                self.population.append(Program(self.img_h, self.img_w, rand_depth, 'full'))
        elif self.init_method == 'grow':
            for _ in range(self.population_size):
                rand_depth = random.randint(self.init_depth[0], self.init_depth[1])
                self.population.append(Program(self.img_h, self.img_w, rand_depth, 'growth'))
        else:  # 'ramped_half_and_half'
            full_num = self.population_size // 2
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
        # clear kernel time in evaluator (if using GPU)
        if self.device in ['numba_cuda', 'py_cuda', 'cupy']:
            self.evaluator.cuda_kernel_time = 0

        # init these params
        training_start = time.time()
        self.cuda_kernel_time = 0
        self.training_time = 0
        self.fitness_evaluation_time = 0
        self.best_program_in_each_gen = []

        # population initialization
        self.population_init()

        # evaluate fitness for the initial population
        fit_eval_start = time.time()
        self.evaluator.evaluate_population(self.population)
        self.fitness_evaluation_time += time.time() - fit_eval_start

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
            fit_eval_start = time.time()
            self.evaluator.evaluate_population(self.population)
            self.fitness_evaluation_time += time.time() - fit_eval_start

            # update
            self._update_generation_properties()
            self._print_population_properties(gen=iter_times)

        # record training time and kernel time (if using GPU)
        self.training_time = time.time() - training_start
        if self.device in ['numba_cuda', 'py_cuda', 'cupy']:
            self.cuda_kernel_time = self.evaluator.cuda_kernel_time

    def run_test(self):
        self.test_evaluator.evaluate_population(self.best_program_in_each_gen)
        self.best_test_program = max(self.best_program_in_each_gen, key=lambda program: program.fitness)
        print(f'[ =========Run Test======== ]')
        print(f'[ Best program in test data ] {self.best_test_program}')
        print(f'[ Accuracy                  ] {self.best_test_program.fitness}')
        print()


# Combine Binary Classifier with Instance Selection Approach.
# Instance Selection (IS) aims to split a whole dataset in to small subsets and train them.
# IS can reduce the computational complexity to speed up GP based image classification.
# Compared with normal GPU classifier, larger classification tasks will see obvious improvements.
class BinaryClassifierWithInstanceSelection(BinaryClassifier):
    """
    Combine Binary Classifier with Instance Selection Approach.
    Instance Selection (IS) aims to split a whole dataset in to small subsets and train them.
    IS can reduce the computational complexity to speed up GP based image classification.
    Compared with normal GPU classifier, larger classification tasks will see obvious improvements.
    """
    def __init__(self, train_set: np.ndarray, train_label: np.ndarray, test_set=None, test_label=None,
                 population_size=500, init_method='ramped_half_and_half', init_depth=(3, 6), max_program_depth=8,
                 generations=50, elist_size=5, tournament_size=7, crossover_prob=0.8, mutation_prob=0.19,
                 device='py_cuda', cu_arch=None, cu_code=None, eval_batch=10, thread_per_block=128):

        super(BinaryClassifierWithInstanceSelection, self).__init__(
            train_set, train_label, test_set, test_label, population_size, init_method, init_depth, max_program_depth,
            generations, elist_size, tournament_size, crossover_prob, mutation_prob, device, cu_arch, cu_code,
            eval_batch, thread_per_block
        )

        # split training set and cores label into 5 subsets
        subset_len = (self.data_size - 1) // 5

        # shuffle train data and train label
        self._shuffle_dataset_and_label()

        # 5 training subsets
        # self.train_subsets = [
        #     train_set[i * subset_len: min((i + 1) * subset_len, len(train_set))]
        #     for i in range(5)
        # ]

        self.train_subsets = [self.train_set[:subset_len],
                              self.train_set[subset_len:2 * subset_len],
                              self.train_set[2 * subset_len:3 * subset_len],
                              self.train_set[3 * subset_len:4 * subset_len],
                              self.train_set[4 * subset_len:]]

        # 5 label subsets
        # self.train_label_subsets = [
        #     train_label[i * subset_len: min((i + 1) * subset_len, len(train_label))]
        #     for i in range(5)
        # ]

        self.train_label_subsets = [self.train_label[:subset_len],
                                    self.train_label[subset_len:2 * subset_len],
                                    self.train_label[2 * subset_len:3 * subset_len],
                                    self.train_label[3 * subset_len:4 * subset_len],
                                    self.train_label[4 * subset_len:]]

    def train(self):
        subset_index = 0
        subset_evaluator = self._get_evaluator(self.train_subsets[subset_index], self.train_label_subsets[subset_index])

        # clear kernel time in evaluator (if using GPU)
        if self.device in ['numba_cuda', 'py_cuda']:
            self.evaluator.cuda_kernel_time = 0

        # init these params
        training_start = time.time()
        self.cuda_kernel_time = 0
        self.training_time = 0
        self.fitness_evaluation_time = 0
        self.best_program_in_each_gen = []

        # population initialization
        self.population_init()

        # evaluate fitness for the initial population
        fit_eval_start = time.time()
        subset_evaluator.evaluate_population(self.population)
        self.fitness_evaluation_time += time.time() - fit_eval_start

        # update
        self._update_generation_properties()
        self._print_population_properties(gen=0)

        # do iterations
        for iter_times in range(1, self.generations):
            new_population: List[Program] = []

            # the index of subset of training set to be evaluated
            cur_subset_index = iter_times // ((self.generations - 1) // 5 + 1)
            if cur_subset_index != subset_index:
                subset_index = cur_subset_index
                del subset_evaluator
                subset_evaluator = self._get_evaluator(self.train_subsets[subset_index],
                                                       self.train_label_subsets[subset_index])

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
            # IS method first evaluates a subset of instances
            fit_eval_start = time.time()
            subset_evaluator.evaluate_population(self.population)
            self.fitness_evaluation_time += time.time() - fit_eval_start

            # evaluate top 10 individuals on the whole training data
            self.population.sort(key=lambda program: program.fitness, reverse=True)
            self.evaluator.evaluate_population(self.population[0: 10])

            # update
            self._update_generation_properties()
            self._print_population_properties(gen=iter_times)

        # record training time and kernel time (if using GPU)
        self.training_time = time.time() - training_start
        if self.device in ['numba_cuda', 'py_cuda']:
            self.cuda_kernel_time = self.evaluator.cuda_kernel_time
