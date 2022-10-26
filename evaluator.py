from typing import List

import numpy as np
from numba import cuda

import device_emp
from device_emp import MAX_TOP, MAX_PROGRAM_LEN
import device
from program import Program


class GPUProgramEvaluator:
    def __init__(self, dataset, label, thread_per_block=64, max_top=10):
        self.dataset = dataset
        self.label = label
        self.data_size = len(dataset)
        self.img_h = len(self.dataset[0])
        self.img_w = len(self.dataset[0][0])
        self.thread_per_block = thread_per_block
        self.max_top = max_top
        #
        self._ddataset = cuda.to_device(self.dataset.reshape(self.data_size, -1).T.reshape(1, -1).squeeze())
        self._dstack = self._allocate_device_stack()
        self._dhist = self._allocate_device_hist_buffer()
        self._dres = self._allocate_device_res_buffer()
        self._dconv_buffer = self._allocate_device_conv_buffer()

    def _allocate_device_stack(self):
        return cuda.device_array(self.data_size * self.img_h * self.img_w, float)

    def _allocate_device_conv_buffer(self):
        return cuda.device_array(self.data_size * self.img_h * self.img_w, float)

    def _allocate_device_hist_buffer(self):
        return cuda.device_array(self.data_size * (device.MAX_PIXEL_VALUE + 1), float)

    def _allocate_device_res_buffer(self):
        return cuda.device_array(self.max_top * self.data_size)

    def fitness_evaluate(self, program: Program):
        name = np.zeros(len(program), int)
        rx = np.zeros(len(program), int)
        ry = np.zeros(len(program), int)
        rh = np.zeros(len(program), int)
        rw = np.zeros(len(program), int)

        for i in range(len(program)):
            name[i] = program[i].name
            if program[i].is_terminal():
                rx[i] = program[i].rx
                ry[i] = program[i].ry
                rh[i] = program[i].rh
                rw[i] = program[i].rw
        name = cuda.to_device(name)
        rx, ry, rh, rw = cuda.to_device(rx), cuda.to_device(ry), cuda.to_device(rh), cuda.to_device(rw)

        grid = int((self.data_size - 1 + self.thread_per_block) / self.thread_per_block)
        device.calc_fit[grid, self.thread_per_block](name, rx, ry, rh, rw, len(program), self.img_h, self.img_w,
                                                     self.data_size, self._ddataset, self._dstack, self._dconv_buffer,
                                                     self._dhist, self._dres)
        cuda.synchronize()
        res = self._dres.copy_to_host()
        correct = 0
        for i in range(self.data_size):
            if self.label[i] > 0 and res[i] > 0 or self.label[i] < 0 and res[i] < 0:
                correct += 1
        program.fitness = correct / self.data_size


class GPUPopulationEvaluator:
    def __init__(self, dataset, label, eval_batch, thread_per_block=64):
        self.dataset = dataset
        self.label = label
        self.data_size = len(dataset)
        self.img_h = len(self.dataset[0])
        self.img_w = len(self.dataset[0][0])
        self.eval_batch = eval_batch
        self.thread_per_block = thread_per_block
        self.max_top = MAX_TOP
        self.max_program_len = MAX_PROGRAM_LEN
        #
        self._ddataset = cuda.to_device(self.dataset.reshape(self.data_size, -1).T.reshape(1, -1).squeeze())
        self._dstack = self._allocate_device_stack()
        self._dhist = self._allocate_device_hist_buffer()
        self._dres = self._allocate_device_res_buffer()
        self._dconv_buffer = self._allocate_device_conv_buffer()
        self.population = ...
        self.pop_size = ...

    def _allocate_device_stack(self):
        return cuda.device_array(self.data_size * self.img_h * self.img_w * self.eval_batch, float)

    def _allocate_device_conv_buffer(self):
        return cuda.device_array(self.data_size * self.img_h * self.img_w * self.eval_batch, float)

    def _allocate_device_hist_buffer(self):
        return cuda.device_array(self.data_size * (device_emp.MAX_PIXEL_VALUE + 1) * self.eval_batch, float)

    def _allocate_device_res_buffer(self):
        return cuda.device_array(self.max_top * self.data_size * self.eval_batch)

    def fitness_evaluate(self, population: List[Program]):
        self.population = population
        self.pop_size = len(population)
        if self.pop_size > self.eval_batch:
            raise RuntimeError('Error: pop size > eval batch.')

        # allocate device side programs
        name = np.zeros((self.pop_size, self.max_program_len), int)
        rx = np.zeros((self.pop_size, self.max_program_len), int)
        ry = np.zeros((self.pop_size, self.max_program_len), int)
        rh = np.zeros((self.pop_size, self.max_program_len), int)
        rw = np.zeros((self.pop_size, self.max_program_len), int)
        plen = np.zeros(self.pop_size, int)

        # parse the program
        for i in range(self.pop_size):
            program = self.population[i]
            plen[i] = len(program)
            for j in range(len(program)):
                name[i][j] = program[j].name
                if program[j].is_terminal():
                    rx[i][j] = program[j].rx
                    ry[i][j] = program[j].ry
                    rh[i][j] = program[j].rh
                    rw[i][j] = program[j].rw

        # copy to device
        name = cuda.to_device(name)
        rx, ry, rh, rw = cuda.to_device(rx), cuda.to_device(ry), cuda.to_device(rh), cuda.to_device(rw)
        plen = cuda.to_device(plen)

        # launch kernel
        grid = (int((self.data_size - 1 + self.thread_per_block) / self.thread_per_block), self.pop_size)
        device_emp.calc_pop_fit[grid, self.thread_per_block](name, rx, ry, rh, rw, plen, self.img_h, self.img_w,
                                                             self.data_size, self._ddataset, self._dstack,
                                                             self._dconv_buffer, self._dhist, self._dres)
        cuda.synchronize()

        # get accuracy
        res = self._dres.copy_to_host().reshape(self.pop_size, -1)
        for i in range(self.pop_size):
            correct = 0
            for j in range(self.data_size):
                if self.label[j] > 0 and res[i][j] > 0 or self.label[j] < 0 and res[i][j] < 0:
                    correct += 1
            self.population[i].fitness = correct / self.data_size
