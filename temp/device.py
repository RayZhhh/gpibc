# This file defines device-side operators.
import math
import os
import random
import time

import numba
import numpy as np
from PIL import Image
from numba import cuda

import fset
from program import Program

MAX_PIXEL_VALUE = 255


# -The conv operation of each image is collaborated by several blocks, which is shown below:
#                       [x x x] [x x x]
#     block(?, 0, 0) => [x x x] [x x x] <= block(?, 0, 1)
#                       [x x x] [x x x]
#
#                       [x x x] [x x x]
#     block(?, 1, 0) => [x x x] [x x x] <= block(?, 1, 1)
#                       [x x x] [x x x]
#     We can learn from the pattern that each thread is responsible for a pixel.


# -Grid size when launching conv kernel:
#     Grid dim: [image_number, (image_height - 1 + BLOCK_H) / BLOCK_H, (image_width - 1 + BLOCK_W) / BLOCK_W]
#     Block dim: [BLOCK_H, BLOCK_W]


# -The dataset is structured as follows:
#                [<--------- img_h * img_w ---------->]
#     image_0 => [000000000000000000000000000000000000]
#     image_1 => [111111111111111111111111111111111111]
#     image_2 => [222222222222222222222222222222222222]


# -The stack is structured as follows:
#              [<--------- img_h * img_w ---------->]
#     top=0 => [000000000000000000000000000000000000]
#              [111111111111111111111111111111111111]
#              [....................................]


@cuda.jit(device=True, inline=True)
def _pixel_ind_in_dataset(x, y, img_h, img_w) -> int:
    img_no = cuda.blockIdx.x
    return img_no * img_h * img_w + img_w * x + y


@cuda.jit(device=True, inline=True)
def _pixel_value_in_dataset(dataset, x, y, img_h, img_w):
    return dataset[_pixel_ind_in_dataset(x, y, img_h, img_w)]


@cuda.jit(device=True, inline=True)
def _pixel_ind_in_stack(x, y, img_h, img_w) -> int:
    img_no = cuda.blockIdx.x
    return img_no * img_h * img_w + img_w * x + y


@cuda.jit(device=True, inline=True)
def _pixel_value_in_stack(stack, x, y, img_h, img_w):
    return stack[_pixel_ind_in_stack(x, y, img_h, img_w)]

#
#              [<-- data_size -->]
#     top=0 => [000000000000000000000000000000000000]
#     top=1 => [111111111111111111111111111111111111]
#              [....................................]
#
@cuda.jit(device=True)
def cu_gstd(res, top, data_size, stack, img_h, img_w, rx, ry, rh, rw):
    if cuda.blockIdx.y == 0 and cuda.blockIdx.z == 0 and cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        mean = 0
        for i in range(rx + rh):
            for j in range(ry + rw):
                mean += _pixel_value_in_stack(stack, i, j, img_h, img_w)
        mean /= rh * rw
        std = 0
        for i in range(rx + rh):
            for j in range(ry + rw):
                value = _pixel_value_in_stack(stack, i, j, img_h, img_w) - mean
                std += value * value
        std /= rh * rw
        std = math.sqrt(std)
        res[data_size * top + cuda.blockIdx.x] = std
    cuda.syncthreads()
        
        
@cuda.jit(device=True)
def cu_sub(res, top, data_size):
    if cuda.blockIdx.y == 0 and cuda.blockIdx.z == 0 and cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        res1 = res[data_size * (top - 2) + cuda.blockIdx.x]
        res2 = res[data_size * (top - 1) + cuda.blockIdx.x]
        res[data_size * (top - 2) + cuda.blockIdx.x] = res1 - res2
    cuda.syncthreads()
        
        
@cuda.jit(device=True)
def cu_region(dataset, stack, img_h, img_w):
    # the pixel coordinates that this thread responsible for
    px = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    if px < img_h and py < img_w:
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = _pixel_value_in_dataset(dataset, px, py, img_h, img_w)
    cuda.syncthreads()


@cuda.jit(device=True)
def cu_lap(stack, img_h, img_w, rx, ry, rh, rw):
    """Args launching or invoking this function:
    Grid dim: [image_number, (image_height - 1 + cuda.blockDim.x) / cuda.blockDim.x, (image_width - 1 + cuda.blockDim.y) / cuda.blockDim.y]
    Block dim: [cuda.blockDim.x, cuda.blockDim.y]

    The Laplacian kernel is: [0, 1, 0]
                             [1,-4, 1]
                             [0, 1, 0].
    """
    # the pixel coordinates that this thread responsible for
    px = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    if rx + 1 <= px < rx + rh - 1 and ry + 1 <= py < ry + rw - 1:
        sum = 0
        sum += _pixel_value_in_stack(stack, px - 1, py, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px, py - 1, img_h, img_w)
        sum -= _pixel_value_in_stack(stack, px, py, img_h, img_w) * 4
        sum += _pixel_value_in_stack(stack, px, py + 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px + 1, py, img_h, img_w)
        sum = max(0, sum)
        sum = min(MAX_PIXEL_VALUE, sum)
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum
    cuda.syncthreads()


@cuda.jit(device=True)
def cu_gau1(stack, img_h, img_w, rx, ry, rh, rw):
    """Args launching or invoking this function:
    Grid dim: [image_number, (image_height - 1 + cuda.blockDim.x) / cuda.blockDim.x, (image_width - 1 + cuda.blockDim.y) / cuda.blockDim.y]
    Block dim: [cuda.blockDim.x, cuda.blockDim.y]

    The Gaussian smooth kernel is: [1, 2, 1]
                                   [2, 4, 2] * (1 / 16).
                                   [1, 2, 1]
    """
    # the pixel coordinates that this thread responsible for
    px = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    if rx + 1 <= px < rx + rh - 1 and ry + 1 <= py < ry + rw - 1:
        sum = 0
        sum += _pixel_value_in_stack(stack, px - 1, py - 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px - 1, py, img_h, img_w) * 2
        sum += _pixel_value_in_stack(stack, px - 1, py + 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px, py - 1, img_h, img_w) * 2
        sum += _pixel_value_in_stack(stack, px, py, img_h, img_w) * 4
        sum += _pixel_value_in_stack(stack, px, py + 1, img_h, img_w) * 2
        sum += _pixel_value_in_stack(stack, px + 1, py - 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px + 1, py, img_h, img_w) * 2
        sum += _pixel_value_in_stack(stack, px + 1, py + 1, img_h, img_w)
        sum = max(0, sum)
        sum = min(MAX_PIXEL_VALUE, sum)
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum / 16
    cuda.syncthreads()


@cuda.jit(device=True)
def cu_sobel_x(stack, img_h, img_w, rx, ry, rh, rw):
    """Args launching or invoking this function:
    Grid dim: [image_number, (image_height - 1 + cuda.blockDim.x) / cuda.blockDim.x, (image_width - 1 + cuda.blockDim.y) / cuda.blockDim.y]
    Block dim: [cuda.blockDim.x, cuda.blockDim.y]

    The Sobel Vertical kernel is: [ 1, 2, 1]
                                  [ 0, 0, 0]
                                  [-1,-2,-1].
    """
    # the pixel coordinates that this thread responsible for
    px = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    if rx + 1 <= px < rx + rh - 1 and ry + 1 <= py < ry + rw - 1:
        sum = 0
        sum += _pixel_value_in_stack(stack, px - 1, py - 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px - 1, py, img_h, img_w) * 2
        sum += _pixel_value_in_stack(stack, px - 1, py + 1, img_h, img_w)
        sum -= _pixel_value_in_stack(stack, px + 1, py - 1, img_h, img_w)
        sum -= _pixel_value_in_stack(stack, px + 1, py, img_h, img_w) * 2
        sum -= _pixel_value_in_stack(stack, px + 1, py + 1, img_h, img_w)
        sum = max(0, sum)
        sum = min(MAX_PIXEL_VALUE, sum)
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum
    cuda.syncthreads()


@cuda.jit(device=True)
def cu_sobel_y(stack, img_h, img_w, rx, ry, rh, rw):
    """Args launching or invoking this function:
    Grid dim: [image_number, (image_height - 1 + cuda.blockDim.x) / cuda.blockDim.x, (image_width - 1 + cuda.blockDim.y) / cuda.blockDim.y]
    Block dim: [cuda.blockDim.x, cuda.blockDim.y]

    The Sobel Horizontal kernel is: [-1, 0, 1 ]
                                    [-2, 0, 2 ]
                                    [-1, 0, 1 ].
    """
    # the pixel coordinates that this thread responsible for
    px = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    if rx + 1 <= px < rx + rh - 1 and ry + 1 <= py < ry + rw - 1:
        sum = 0
        sum -= _pixel_value_in_stack(stack, px - 1, py - 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px - 1, py + 1, img_h, img_w)
        sum -= _pixel_value_in_stack(stack, px, py - 1, img_h, img_w) * 2
        sum += _pixel_value_in_stack(stack, px, py + 1, img_h, img_w) * 2
        sum -= _pixel_value_in_stack(stack, px + 1, py - 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px + 1, py + 1, img_h, img_w)
        sum = max(0, sum)
        sum = min(MAX_PIXEL_VALUE, sum)
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum
    cuda.syncthreads()


@cuda.jit(device=True)
def cu_log1(stack, img_h, img_w, rx, ry, rh, rw):
    """Args launching or invoking this function:
    Grid dim: [image_number, (image_height - 1 + cuda.blockDim.x) / cuda.blockDim.x, (image_width - 1 + cuda.blockDim.y) / cuda.blockDim.y]
    Block dim: [cuda.blockDim.x, cuda.blockDim.y]

    The LoG kernel is: [0,  0,  1,  0,  0]
                       [0,  1,  2,  1,  0]
                       [1,  2,-16,  2,  1]
                       [0,  1,  2,  1,  0]
                       [0,  0,  1,  0,  0].
    """
    # the pixel coordinates that this thread responsible for
    px = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    if rx + 2 <= px < rx + rh - 2 and ry + 2 <= py < ry + rw - 2:
        sum = 0
        sum += _pixel_value_in_stack(stack, px - 2, py, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px - 1, py - 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px - 1, py, img_h, img_w) * 2
        sum += _pixel_value_in_stack(stack, px - 1, py + 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px, py - 2, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px, py - 1, img_h, img_w) * 2
        sum -= _pixel_value_in_stack(stack, px, py, img_h, img_w) * 16
        sum += _pixel_value_in_stack(stack, px, py + 1, img_h, img_w) * 2
        sum += _pixel_value_in_stack(stack, px, py + 2, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px + 1, py - 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px + 1, py, img_h, img_w) * 2
        sum += _pixel_value_in_stack(stack, px + 1, py + 1, img_h, img_w)
        sum += _pixel_value_in_stack(stack, px + 2, py, img_h, img_w)
        sum = max(0, sum)
        sum = min(MAX_PIXEL_VALUE, sum)
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum
    cuda.syncthreads()


@cuda.jit(device=True)
def cu_lbp(stack, img_h, img_w, rx, ry, rh, rw):
    """Args launching or invoking this function:
    Grid dim: [image_number, (image_height - 1 + cuda.blockDim.x) / cuda.blockDim.x, (image_width - 1 + cuda.blockDim.y) / cuda.blockDim.y]
    Block dim: [cuda.blockDim.x, cuda.blockDim.y]

    Step 1:
        calculate the value of each pixel based on the threshold
        pixel_lbp(i) = 0 if pixel(i) < center else 1

    Step 2:
        calculate the value of the center pixel using the weights: [  1,  2,  4]
                                                                   [128,  C,  8]
                                                                   [ 64, 32, 16]
    """
    # the pixel coordinates that this thread responsible for
    px = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    if rx + 1 <= px < rx + rh - 1 and ry + 1 <= py < ry + rw - 1:
        sum = 0
        cp_value = _pixel_value_in_stack(stack, px, py, img_h, img_w)
        p_1 = _pixel_value_in_stack(stack, px - 1, py - 1, img_h, img_w)
        p_2 = _pixel_value_in_stack(stack, px - 1, py, img_h, img_w)
        p_4 = _pixel_value_in_stack(stack, px - 1, py + 1, img_h, img_w)
        p_8 = _pixel_value_in_stack(stack, px, py + 1, img_h, img_w)
        p_16 = _pixel_value_in_stack(stack, px + 1, py + 1, img_h, img_w)
        p_32 = _pixel_value_in_stack(stack, px + 1, py, img_h, img_w)
        p_64 = _pixel_value_in_stack(stack, px + 1, py - 1, img_h, img_w)
        p_128 = _pixel_value_in_stack(stack, px, py - 1, img_h, img_w)
        # add up based on the weights
        sum += p_1 if p_1 >= cp_value else 0
        sum += p_2 * 2 if p_2 >= cp_value else 0
        sum += p_4 * 4 if p_4 >= cp_value else 0
        sum += p_8 * 8 if p_8 >= cp_value else 0
        sum += p_16 * 16 if p_16 >= cp_value else 0
        sum += p_32 * 32 if p_32 >= cp_value else 0
        sum += p_64 * 64 if p_64 >= cp_value else 0
        sum += p_128 * 128 if p_128 >= cp_value else 0
        sum = max(0, sum)
        sum = min(MAX_PIXEL_VALUE, sum)
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum
    cuda.syncthreads()


# -Histogram based operators.
# -The histogram of each image stores in the histogram buffer is structured as follows:
#                [<------ MAX_PIXEL_VALUE + 1 ------->]
#     image_0 => [000000000000000000000000000000000000]
#     image_1 => [111111111111111111111111111111111111]
#     image_2 => [222222222222222222222222222222222222]
#                [....................................]


@cuda.jit(device=True, inline=True)
def _hist_buffer_ind(value) -> int:
    img_no = cuda.blockIdx.x
    return img_no * (MAX_PIXEL_VALUE + 1) + value


@cuda.jit(device=True, inline=True)
def _hist_buffer_value(hist_buffer, value):
    return hist_buffer[_hist_buffer_ind(value)]


@cuda.jit(device=True)
def cu_hist_eq_statistic(stack, hist_buffer, img_h, img_w, rx, ry, rh, rw):
    """Args launching or invoking this function:
    Grid dim: [image_number]
    Block dim: [1]
    """
    if cuda.blockIdx.y == 0 and cuda.blockIdx.z == 0 and cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        # clear the hist buffer
        for i in range(0, MAX_PIXEL_VALUE + 1):
            hist_buffer[_hist_buffer_ind(i)] = 0

        # if cuda.blockIdx.y == 0 and cuda.blockIdx.z == 0 and cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                value = int(_pixel_value_in_stack(stack, i, j, img_h, img_w))
                value = max(value, 0)
                value = min(value, MAX_PIXEL_VALUE)
                hist_buffer[_hist_buffer_ind(value)] += 1 / (rh * rw)

        for i in range(1, MAX_PIXEL_VALUE + 1):
            hist_buffer[_hist_buffer_ind(i)] += hist_buffer[_hist_buffer_ind(i - 1)]

        for i in range(MAX_PIXEL_VALUE + 1):
            hist_buffer[_hist_buffer_ind(i)] *= MAX_PIXEL_VALUE
    cuda.syncthreads()


@cuda.jit(device=True)
def cu_hist_eq_update_pixel(stack, hist_buffer, img_h, img_w, rx, ry, rh, rw):
    """Args launching or invoking this function:
    Grid dim: [image_number, (image_height - 1 + cuda.blockDim.x) / cuda.blockDim.x, (image_width - 1 + cuda.blockDim.y) / cuda.blockDim.y]
    Block dim: [cuda.blockDim.x, cuda.blockDim.y]
    """
    # the pixel coordinates that this thread responsible for
    px = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    if rx <= px < rx + rh and ry <= py < ry + rw:
        value = int(_pixel_value_in_stack(stack, px, py, img_h, img_w))
        value = max(value, 0)
        value = min(value, MAX_PIXEL_VALUE)
        new_value = _hist_buffer_value(hist_buffer, value)
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = new_value
    cuda.syncthreads()
    
    
@cuda.jit()
def program_fit_eval(nlen, name, rx, ry, rh, rw, stack, hist_buffer, res, im_h, im_w, dataset, data_size):
    region_x, region_y, region_h, region_w = 0, 0, 0, 0
    top = 0
    for i in range(nlen - 1, -1, -1):
        if name[i] == fset.Region_R:
            region_x, region_y, region_h, region_w = rx[i], ry[i], rh[i], rw[i]
            cu_region(dataset, stack, im_h, im_w)
        elif name[i] == fset.Region_S:
            region_x, region_y, region_h, region_w = rx[i], ry[i], rh[i], rw[i]
            cu_region(dataset, stack, im_h, im_w)
        elif name[i] == fset.G_Std:
            cu_gstd(res, top, data_size, stack, im_h, im_w, region_x, region_y, region_h, region_w)
            top += 1
        elif name[i] == fset.Hist_Eq:
            cu_hist_eq_statistic(stack, hist_buffer, im_h, im_w, region_x, region_y, region_h, region_w)
            cu_hist_eq_update_pixel(stack, hist_buffer, im_h, im_w, region_x, region_y, region_h, region_w)
        elif name[i] == fset.Gau1:
            cu_gau1(stack, im_h, im_w, region_x, region_y, region_h, region_w)
            region_x, region_y, region_h, region_w = region_x + 1, region_y + 1, region_h - 2, region_w - 2
        elif name[i] == fset.Gau11:
            print('Currently unsupported')
        elif name[i] == fset.GauXY:
            print('Currently unsupported')
        elif name[i] == fset.Lap:
            cu_lap(stack, im_h, im_w, region_x, region_y, region_h, region_w)
            region_x, region_y, region_h, region_w = region_x + 1, region_y + 1, region_h - 2, region_w - 2
        elif name[i] == fset.Sobel_X:
            cu_sobel_x(stack, im_h, im_w, region_x, region_y, region_h, region_w)
            region_x, region_y, region_h, region_w = region_x + 1, region_y + 1, region_h - 2, region_w - 2
        elif name[i] == fset.Sobel_Y:
            cu_sobel_y(stack, im_h, im_w, region_x, region_y, region_h, region_w)
            region_x, region_y, region_h, region_w = region_x + 1, region_y + 1, region_h - 2, region_w - 2
        elif name[i] == fset.LoG1:
            cu_log1(stack, im_h, im_w, region_x, region_y, region_h, region_w)
            region_x, region_y, region_h, region_w = region_x + 2, region_y + 2, region_h - 4, region_w - 4
        elif name[i] == fset.LoG2:
            print('Do not support LoG2')
        elif name[i] == fset.HOG:
            print('Do not support HOG')
        elif name[i] == fset.Sub:
            cu_sub(res, top, data_size)
            top -= 1
        else:
            print('Error node type: ', name[i])
    if not top == 1:
        print('error top != 1 ===', top)


MAX_TOP = 6
BLOCK_H = 16  # equals to blockDim.x
BLOCK_W = 16  # equals to blockDim.y
class GPUProgramEvaluator:
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
        self.data_size = len(dataset)
        self.img_h = len(self.dataset[0])
        self.img_w = len(self.dataset[0][0])
        #
        self._ddataset = cuda.to_device(self.dataset.reshape(1, -1).squeeze())
        self._dstack = self._allocate_device_stack()
        self._dhist = self._allocate_device_hist_buffer()
        self._dres = self._allocate_device_res_buffer()

    def _allocate_device_stack(self):
        return cuda.device_array(self.data_size * self.img_h * self.img_w)

    def _allocate_device_hist_buffer(self):
        return cuda.device_array(self.data_size * (MAX_PIXEL_VALUE + 1))

    def _allocate_device_res_buffer(self):
        return cuda.device_array(MAX_TOP * self.data_size)

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
        rx = cuda.to_device(rx)
        ry = cuda.to_device(ry)
        rh = cuda.to_device(rh)
        rw = cuda.to_device(rw)

        grid = (self.data_size, int((self.img_h - 1 + BLOCK_H) / BLOCK_H), int((self.img_w - 1 + BLOCK_W) / BLOCK_W))
        block = (BLOCK_H, BLOCK_W)
        program_fit_eval[grid, block](len(program), name, rx, ry, rh, rw, self._dstack, self._dhist, self._dres,
                                      self.img_h, self.img_w, self._ddataset, self.data_size)
        cuda.synchronize()
        res = self._dres.copy_to_host()
        correct = 0
        for i in range(self.data_size):
            if self.label[i] > 0 and res[i] > 0 or self.label[i] < 0 and res[i] < 0:
                correct += 1
        program.fitness = correct / self.data_size


def create_dataset():
    data_ret = np.array([])
    label_ret = np.array([])
    for root, ds, fs in os.walk('../jaffe/happiness'):
        num = 0
        for f in fs:
            image = Image.open('jaffe/happiness/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((128, 128))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [1])

            num += 1
            # if num >= 10:
            #     break

    for root, ds, fs in os.walk('../jaffe/surprise'):
        num = 0
        for f in fs:
            image = Image.open('jaffe/surprise/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((128, 128))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [-1])

            num += 1
            # if num >= 10:
            #     break

    return data_ret.reshape(-1, 128, 128), label_ret


if __name__ == '__main__':
    data, label = create_dataset()
    ts = time.time()
    for i in range(500):
        program = Program(128, 128, init_method='growth')
        # print(program)
        fit = GPUProgramEvaluator(data, label)
        fit.fitness_evaluate(program)
        # print(program.fitness)
    print('pop time: ', time.time() - ts)