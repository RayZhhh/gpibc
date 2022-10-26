import math
import random
import sys
from typing import List

import numba
from numba import cuda
import numpy as np

import fset
from program import Program
from tree import Node

THREAD_PER_BLOCK = 32

MAX_PIXEL_VALUE = 255


def program_to_device(program: Program):
    node = np.zeros(len(program), int)
    rx = np.zeros(len(program), int)
    ry = np.zeros(len(program), int)
    rh = np.zeros(len(program), int)
    rw = np.zeros(len(program), int)

    for i in range(len(program)):
        node[i] = program[i].name
        if program[i].is_terminal():
            rx[i] = program[i].rx
            ry[i] = program[i].ry
            rh[i] = program[i].rh
            rw[i] = program[i].rw

    node = cuda.to_device(node)
    rx = cuda.to_device(rx)
    ry = cuda.to_device(ry)
    rh = cuda.to_device(rh)
    rw = cuda.to_device(rw)
    return node, rx, ry, rh, rw


# -The device side dataset is structured as follows:
#                [0 0 0]  [1 1 1]  [2 2 2]                   [0 0 0 0 0 0 0 0 0]
#     raw image: [0 0 0]  [1 1 1]  [2 2 2] ... => reshape => [1 1 1 1 1 1 1 1 1] => transpose =>
#                [0 0 0]  [1 1 1]  [2 2 2]                   [2 2 2 2 2 2 2 2 2]
#
#              [<---- data_size ---->]           [<---- data_size ---->]
#              [0 1 2 3 4 5 6 7 8 9 .]           [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#              [0 1 2 3 4 5 6 7 8 9 .]           [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#              [0 1 2 3 4 5 6 7 8 9 .]           [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#              [0 1 2 3 4 5 6 7 8 9 .]           [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#     dataset: [0 1 2 3 4 5 6 7 8 9 .]    stack: [0 1 2 3 4 5 6 7 8 9 .]    thread => [0]    thread => [1]
#              [0 1 2 3 4 5 6 7 8 9 .]           [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#              [0 1 2 3 4 5 6 7 8 9 .]           [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#              [0 1 2 3 4 5 6 7 8 9 .]           [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#              [0 1 2 3 4 5 6 7 8 9 .]           [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]


# -Also a buffer for storing temp conv value is allocated
#                  [<---- data_size ---->]
#                  [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#                  [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#                  [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#                  [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#     conv buffer: [0 1 2 3 4 5 6 7 8 9 .]    thread => [0]    thread => [1]
#                  [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#                  [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#                  [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#                  [0 1 2 3 4 5 6 7 8 9 .]              [0]              [1]
#     In this approach, each thread is responsible for an image.


# -Grid size when launch kernels:
#     Grid: int((data_size -1 + THREAD_PER_BLOCK) / THREAD_PER_BLOCK)


@cuda.jit(device=True, inline=True)
def _pixel_dataset_index(data_size, im_w, i, j) -> int:
    pixel_row = i * im_w + j
    pixel_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return pixel_row * data_size + pixel_col


def _pixel_value_in_dataset(dataset, data_size, im_w, i, j):
    return dataset[_pixel_dataset_index(data_size, im_w, i, j)]


@cuda.jit(device=True, inline=True)
def _pixel_stack_index(data_size, im_w, i, j) -> int:
    pixel_row = i * im_w + j
    pixel_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return pixel_row * data_size + pixel_col


@cuda.jit(device=True, inline=True)
def _pixel_value_in_stack(stack, data_size, im_w, i, j):
    return stack[_pixel_stack_index(data_size, im_w, i, j)]


@cuda.jit(device=True, inline=True)
def _pixel_conv_buffer_index(data_size, im_w, i, j) -> int:
    """Get the index in the conv buffer.
    We pre-allocate an array with the same size of dataset for the conv operation.
    """
    img_no = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pixel_index_in_img = i * im_w + j
    pixel_index_in_buffer = pixel_index_in_img * data_size + img_no
    return pixel_index_in_buffer


@cuda.jit(device=True, inline=True)
def _intensity_hist_buffer_index(data_size, intensity) -> int:
    img_no = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return intensity * data_size + img_no


@cuda.jit(device=True)
def sub(stack, minuend_top, subtrahend_top, data_size, im_h, im_w, res_top):
    """Sub"""
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        # the std0 locates at [0, 0]
        std0_index = _pixel_stack_index(data_size, minuend_top, im_w, 0, 0)
        std1_index = _pixel_stack_index(data_size, subtrahend_top, im_w, 0, 0)
        res_index = _pixel_stack_index(data_size, res_top, im_w, 0, 0)
        stack[res_index] = stack[std0_index] - stack[std1_index]

    cuda.syncthreads()




@cuda.jit(device=True)
def hist_eq(stack, cur_top, data_size, im_h, im_w, res_top, rx, ry, rh, rw, hist_buffer):
    """Performing Historical Equalisation to the input region."""
    condition = rx + rh <= im_h and ry + rw <= im_w

    if not condition:
        print('Error: Do not satisfy the condition: rx + rh <= im_h and ry + rw <= im_w')

    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:

        # clear the buffer
        for i in range(0, MAX_PIXEL_VALUE + 1):
            hist_buffer[_intensity_hist_buffer_index(data_size, i - 1)] = 0

        # statistic intensity of each pixel, this process can not perform coalesced access
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                pixel_index = _pixel_stack_index(data_size, cur_top, im_w, i, j)
                pixel_value = int(stack[pixel_index])
                buffer_index = _intensity_hist_buffer_index(data_size, pixel_value)
                hist_buffer[buffer_index] += 1

        # add up
        for i in range(1, MAX_PIXEL_VALUE + 1):
            hist_buffer[_intensity_hist_buffer_index(data_size, i)] += \
                hist_buffer[_intensity_hist_buffer_index(data_size, i - 1)]

        # mapping table
        for i in range(0, MAX_PIXEL_VALUE + 1):
            hist_buffer[_intensity_hist_buffer_index(data_size, i)] /= (rh * rw)
            hist_buffer[_intensity_hist_buffer_index(data_size, i)] *= (MAX_PIXEL_VALUE - 1)

        # update the intensity value of each pixel of the region
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                raw_pixel_intensity = stack[_pixel_stack_index(data_size, res_top, im_w, i, j)]
                new_pixel_intensity = hist_buffer[_intensity_hist_buffer_index(data_size, int(raw_pixel_intensity))]
                stack[_pixel_stack_index(data_size, res_top, im_w, i, j)] = new_pixel_intensity

    cuda.syncthreads()


@cuda.jit(device=True)
def region(dataset, data_size, im_h, im_w, stack, res_top):
    """Both Region_S and Region_R execute this function,
    the only difference is the region size which is stored in the local memory.
    This function copy the image from dataset to stack.
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(im_h):
            for j in range(im_w):
                dataset_index = _pixel_dataset_index(data_size, im_w, i, j)
                res_stack_index = _pixel_stack_index(data_size, res_top, im_w, i, j)
                stack[res_stack_index] = dataset[dataset_index]

    cuda.syncthreads()


@cuda.jit(device=True)
def g_std(stack, cur_top, data_size, im_h, im_w, res_top, rx, ry, rh, rw):
    """Calculate the standard deviation of the region."""

    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:

        avg = 0
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                avg += stack[_pixel_stack_index(data_size, cur_top, im_w, i, j)]
        # for i in range(0, im_h):
        #     for j in range(0, im_w):
        #         avg += stack[_pixel_stack_index(data_size, cur_top, im_w, i, j)]

        avg /= rh * rw

        deviation = 0
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                value = stack[_pixel_stack_index(data_size, cur_top, im_w, i, j)] - avg
                deviation += value * value
        # for i in range(0, im_h):
        #     for j in range(0, im_w):
        #         deviation += (stack[_pixel_stack_index(data_size, cur_top, im_w, i, j)] - avg) ** 2

        deviation /= rh * rw
        deviation = math.sqrt(deviation)
        stack_index = _pixel_stack_index(data_size, res_top, im_w, 0, 0)
        stack[stack_index] = deviation

    cuda.syncthreads()


@cuda.jit(device=True)
def lap(stack, data_size, im_w, rx, ry, rh, rw, buffer):
    """Achieve parallel Lap function.
    In this implementation, a thread is responsible for an image. This function uses top + 1 as buffer
    After Lap operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Laplacian kernel is: [0, 1, 0]
                             [1,-4, 1]
                             [0, 1, 0].
    """

    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        # calculate each pixel result, store them in top + 1 addresses as buffer
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                index0 = _pixel_stack_index(data_size, cur_top, im_w, i, j)
                index1 = _pixel_stack_index(data_size, cur_top, im_w, i, j - 1)
                index2 = _pixel_stack_index(data_size, cur_top, im_w, i, j + 1)
                index3 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j)
                index4 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j)
                pixel_res = -4 * stack[index0] + stack[index1] + stack[index2] + stack[index3] + stack[index4]
                pixel_res = pixel_res if pixel_res >= 0 else 0
                pixel_res = pixel_res if pixel_res <= MAX_PIXEL_VALUE else MAX_PIXEL_VALUE

                # store to stack
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                buffer[buffer_index] = pixel_res

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                stack_index_target = _pixel_stack_index(data_size, res_top, im_w, i, j)
                stack[stack_index_target] = buffer[buffer_index]

    cuda.syncthreads()


@cuda.jit(device=True)
def gau1(stack, cur_top, data_size, im_h, im_w, res_top, rx, ry, rh, rw, buffer):
    """Achieve parallel Gau1 function
    In this implementation, a thread is responsible for an image.
    After Gau1 operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Gaussian smooth kernel is: [1, 2, 1]
                                   [2, 4, 2] * (1 / 16).
                                   [1, 2, 1]
    """
    condition = rx + rh <= im_h and ry + rw <= im_w

    if not condition:
        print('Error: Do not satisfy the condition: rx + rh <= im_h and ry + rw <= im_w')

    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                # get conv result of pixel i, j
                index0 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j - 1)
                index1 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j)
                index2 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j + 1)
                index3 = _pixel_stack_index(data_size, cur_top, im_w, i, j - 1)
                index4 = _pixel_stack_index(data_size, cur_top, im_w, i, j)
                index5 = _pixel_stack_index(data_size, cur_top, im_w, i, j + 1)
                index6 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j - 1)
                index7 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j)
                index8 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j + 1)
                pixel_res = stack[index0] + 2 * stack[index1] + stack[index2] + \
                            2 * stack[index3] + 4 * stack[index4] + 2 * stack[index5] + \
                            stack[index6] + 2 * stack[index7] + stack[index8]
                pixel_res /= 16
                pixel_res = pixel_res if pixel_res >= 0 else 0
                pixel_res = pixel_res if pixel_res <= MAX_PIXEL_VALUE else MAX_PIXEL_VALUE

                # store to stack
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                buffer[buffer_index] = pixel_res

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                stack_index_target = _pixel_stack_index(data_size, res_top, im_w, i, j)
                stack[stack_index_target] = buffer[buffer_index]

    cuda.syncthreads()


@cuda.jit(device=True)
def sobel_x(stack, cur_top, data_size, im_h, im_w, res_top, rx, ry, rh, rw, buffer):
    """Perform Sobel_X on image.
    In this implementation, a thread is responsible for an image.
    After Sobel_X operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Sobel Vertical kernel is: [ 1, 2, 1]
                                  [ 0, 0, 0]
                                  [-1,-2,-1].
    """
    condition = rx + rh <= im_h and ry + rw <= im_w

    if not condition:
        print('Error: Do not satisfy the condition: rx + rh <= im_h and ry + rw <= im_w')

    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                # get conv result of pixel i, j
                index0 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j - 1)
                index1 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j)
                index2 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j + 1)
                index3 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j - 1)
                index4 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j)
                index5 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j + 1)
                pixel_res = stack[index0] + 2 * stack[index1] + stack[index2] - \
                            stack[index3] - 2 * stack[index4] - stack[index5]
                pixel_res = pixel_res if pixel_res >= 0 else 0
                pixel_res = pixel_res if pixel_res <= MAX_PIXEL_VALUE else MAX_PIXEL_VALUE

                # store to stack
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                buffer[buffer_index] = pixel_res

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                stack_index_target = _pixel_stack_index(data_size, res_top, im_w, i, j)
                stack[stack_index_target] = buffer[buffer_index]

    cuda.syncthreads()


@cuda.jit(device=True)
def sobel_y(stack, cur_top, data_size, im_h, im_w, res_top, rx, ry, rh, rw, buffer):
    """Perform Sobel_Y on image.
    In this implementation, a thread is responsible for an image.
    After Sobel_Y operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Sobel Horizontal kernel is: [-1, 0, 1 ]
                                    [-2, 0, 2 ]
                                    [-1, 0, 1 ].
    """
    condition = rx + rh <= im_h and ry + rw <= im_w

    if not condition:
        print('Error: Do not satisfy the condition: rx + rh <= im_h and ry + rw <= im_w')

    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                # get conv result of pixel i, j
                index0 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j - 1)
                index1 = _pixel_stack_index(data_size, cur_top, im_w, i, j - 1)
                index2 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j - 1)
                index3 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j + 1)
                index4 = _pixel_stack_index(data_size, cur_top, im_w, i, j + 1)
                index5 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j + 1)
                pixel_res = - 1 * stack[index0] - 2 * stack[index1] - stack[index2] + \
                            stack[index3] + 2 * stack[index4] + stack[index5]
                pixel_res = pixel_res if pixel_res >= 0 else 0
                pixel_res = pixel_res if pixel_res <= MAX_PIXEL_VALUE else MAX_PIXEL_VALUE

                # store to stack
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                buffer[buffer_index] = pixel_res

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                stack_index_target = _pixel_stack_index(data_size, res_top, im_w, i, j)
                stack[stack_index_target] = buffer[buffer_index]

    cuda.syncthreads()


@cuda.jit(device=True)
def log1(stack, cur_top, data_size, im_h, im_w, res_top, rx, ry, rh, rw, buffer):
    """Perform LoG1 on image.
        In this implementation, a thread is responsible for an image.
        After LoG1 operation, rx += 2; ry += 2; rh -= 4; rw -= 4.

        The LoG kernel is: [0,  0,  1,  0,  0]
                           [0,  1,  2,  1,  0]
                           [1,  2,-16,  2,  1]
                           [0,  1,  2,  1,  0]
                           [0,  0,  1,  0,  0].
    """
    condition = rx + rh <= im_h and ry + rw <= im_w

    if not condition:
        print('Error: Do not satisfy the condition: rx + rh <= im_h and ry + rw <= im_w')

    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 2, rx + rh - 2):
            for j in range(ry + 2, ry + rw - 2):
                # get conv result of pixel i, j
                index0 = _pixel_stack_index(data_size, cur_top, im_w, i - 2, j)
                index1 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j - 1)
                index2 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j)
                index3 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j + 1)
                index4 = _pixel_stack_index(data_size, cur_top, im_w, i, j - 2)
                index5 = _pixel_stack_index(data_size, cur_top, im_w, i, j - 1)
                index6 = _pixel_stack_index(data_size, cur_top, im_w, i, j)
                index7 = _pixel_stack_index(data_size, cur_top, im_w, i, j + 1)
                index8 = _pixel_stack_index(data_size, cur_top, im_w, i, j + 2)
                index9 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j - 1)
                index10 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j)
                index11 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j + 1)
                index12 = _pixel_stack_index(data_size, cur_top, im_w, i + 2, j)
                pixel_res = stack[index0] + stack[index1] + 2 * stack[index2] + stack[index3] + \
                            stack[index4] + 2 * stack[index5] - 16 * stack[index6] + 2 * stack[index7] + stack[index8] + \
                            stack[index9] + 2 * stack[index10] + stack[index11] + stack[index12]
                pixel_res = pixel_res if pixel_res >= 0 else 0
                pixel_res = pixel_res if pixel_res <= MAX_PIXEL_VALUE else MAX_PIXEL_VALUE

                # store to stack
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                buffer[buffer_index] = pixel_res

        # copy the result from buffer to stack
        for i in range(rx + 2, rx + rh - 2):
            for j in range(ry + 2, ry + rw - 2):
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                stack_index_target = _pixel_stack_index(data_size, res_top, im_w, i, j)
                stack[stack_index_target] = buffer[buffer_index]

    cuda.syncthreads()


@cuda.jit(device=True)
def hog(stack, cur_top, data_size, im_h, im_w, res_top, rx, ry, rh, rw, buffer):
    """Histogram of Oriented Gradients"""
    pass


@cuda.jit(device=True)
def lbp(stack, cur_top, data_size, im_h, im_w, res_top, rx, ry, rh, rw, buffer):
    """Perform Local Binary Pattern operation to images.
    Step 1:
        calculate the value of each pixel based on the threshold
        pixel_lbp(i) = 0 if pixel(i) < center else 1

    Step 2:
        calculate the value of the center pixel using the weights: [  1,  2,  4]
                                                                   [128,  C,  8]
                                                                   [ 64, 32, 16]
    """
    condition = rx + rh <= im_h and ry + rw <= im_w

    if not condition:
        print('Error: Do not satisfy the condition: rx + rh <= im_h and ry + rw <= im_w')

    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:

        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                index0 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j - 1)
                index1 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j)
                index2 = _pixel_stack_index(data_size, cur_top, im_w, i - 1, j + 1)
                index3 = _pixel_stack_index(data_size, cur_top, im_w, i, j - 1)
                index4 = _pixel_stack_index(data_size, cur_top, im_w, i, j + 1)
                index5 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j - 1)
                index6 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j)
                index7 = _pixel_stack_index(data_size, cur_top, im_w, i + 1, j + 1)
                center = stack[_pixel_stack_index(data_size, cur_top, im_w, i, j)]
                center_pixel_res = 0
                center_pixel_res += 1 if stack[index0] >= center else 0
                center_pixel_res += 2 if stack[index1] >= center else 0
                center_pixel_res += 4 if stack[index2] >= center else 0
                center_pixel_res += 128 if stack[index3] >= center else 0
                center_pixel_res += 8 if stack[index4] >= center else 0
                center_pixel_res += 64 if stack[index5] >= center else 0
                center_pixel_res += 32 if stack[index6] >= center else 0
                center_pixel_res += 16 if stack[index7] >= center else 0
                center_pixel_res = center_pixel_res if center_pixel_res <= MAX_PIXEL_VALUE else MAX_PIXEL_VALUE
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                buffer[buffer_index] = center_pixel_res

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                buffer_index = _pixel_conv_buffer_index(data_size, im_w, i, j)
                stack_index_target = _pixel_stack_index(data_size, res_top, im_w, i, j)
                stack[stack_index_target] = buffer[buffer_index]

    cuda.syncthreads()


@cuda.jit()
def calc_fit(node_type, rx, ry, rh, rw, program_len, img_h, img_w, data_size, dataset, stack,
             conv_buffer, hist_buffer, label_set, result_arr):
    top = 0
    reg_x, reg_y, reg_h, reg_w = 0, 0, 0, 0

    # reverse iteration
    for i in range(program_len - 1, -1, -1):
        if node_type[i] == fset.Region_R:
            reg_x = rx[i]
            reg_y = ry[i]
            reg_h = rh[i]
            reg_w = rw[i]
            region(dataset, data_size, img_h, img_w, stack, top)
        elif node_type[i] == fset.Region_S:
            reg_x = rx[i]
            reg_y = ry[i]
            reg_h = rh[i]
            reg_w = rw[i]
            region(dataset, data_size, img_h, img_w, stack, top)
        elif node_type[i] == fset.G_Std:
            g_std(stack, top, data_size, img_h, img_w, top, reg_x, reg_y, reg_h, reg_w)
            top += 1
        elif node_type[i] == fset.Hist_Eq:
            hist_eq(stack, top, data_size, img_h, img_w, top, reg_x, reg_y, reg_h, reg_w, hist_buffer)
        elif node_type[i] == fset.Gau1:
            gau1(stack, top, data_size, img_h, img_w, top, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x += 1
            reg_y += 1
            reg_h -= 2
            reg_w -= 2
        elif node_type[i] == fset.Gau11:
            print('Currently unsupported')
        elif node_type[i] == fset.GauXY:
            print('Currently unsupported')
        elif node_type[i] == fset.Lap:
            lap(stack, top, data_size, img_h, img_w, top, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x += 1
            reg_y += 1
            reg_h -= 2
            reg_w -= 2
        elif node_type[i] == fset.Sobel_X:
            sobel_x(stack, top, data_size, img_h, img_w, top, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x += 1
            reg_y += 1
            reg_h -= 2
            reg_w -= 2
        elif node_type[i] == fset.Sobel_Y:
            sobel_y(stack, top, data_size, img_h, img_w, top, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x += 1
            reg_y += 1
            reg_h -= 2
            reg_w -= 2
        elif node_type[i] == fset.LoG1:
            log1(stack, top, data_size, img_h, img_w, top, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x += 2
            reg_y += 2
            reg_h -= 4
            reg_w -= 4
        elif node_type[i] == fset.LoG2:
            print('Do not support LoG2')
        elif node_type[i] == fset.HOG:
            print('Do not support HOG')
        elif node_type[i] == fset.Sub:
            sub(stack, top - 2, top - 1, data_size, img_h, img_w, top - 2)
            top -= 1
        else:
            print('Error node type: ', node_type[i])

    # compare the predicted value with the label
    # set the corresponding item to 1 if the prediction is correct
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if img_index < data_size:
        predict = stack[_pixel_stack_index(data_size, 0, img_w, 0, 0)]
        label = label_set[img_index]
        if predict < 0 and label < 0 or predict > 0 and label > 0:
            result_arr[img_index] = 1

    if top != 1:
        print('Error: top != 1, which may caused by an illegal prefix of the program.')

    cuda.syncthreads()


def eval_program_fitness_gpu(program: Program, img_h, img_w, data_size, dataset, stack,
                             conv_buffer, hist_buffer, label_set):
    # copy the program to device
    node_type, node_rx, node_ry, node_rh, node_rw = program_to_device(program)
    # allocate result
    result = allocate_device_result(data_size)
    # lunch kernel
    calc_fit[int(data_size / cuda.blockDim.x) + 1, cuda.blockDim.x](
        node_type, node_rx, node_ry, node_rh, node_rw, len(program), img_h, img_w, data_size, dataset, stack,
        conv_buffer, hist_buffer, label_set, result
    )
    cuda.synchronize()
    # copy back result
    result = result.copy_to_host()
    # cal fitness
    program.fitness = result.sum() / data_size


@cuda.jit()
def test_kernel(img_h, img_w, data_size, dataset, stack, conv_buffer):
    region(dataset, data_size, img_h, img_w, stack, 0)
    lbp(stack, 0, data_size, img_h, img_w, 0, 0, 0, 128, 128, conv_buffer)


@cuda.jit()
def empty_kernel():
    print('empty kernel')


if __name__ == '__main__':
    import numpy as np
    import time

    img_h, img_w = 128, 128
    data_size = 5000

    print('start test...')

    dataset = np.ones(img_h * img_w * data_size, float)
    # for i in range(0, len(dataset)):
    #     dataset[i] = random.random()
    # dataset = cuda.to_device(dataset)

    label = np.ones(data_size, int)
    label = cuda.to_device(label)


    conv_buffer = allocate_device_conv_buffer(img_h, img_w, data_size)
    stack = allocate_device_side_stack(img_h, img_w, data_size)
    # hist_buffer = allocate_device_hist_buffer(data_size)

    ts = time.time()
    test_kernel[int(data_size / cuda.blockDim.x) + 1, cuda.blockDim.x](
        img_h, img_w, data_size, dataset, stack, conv_buffer
    )
    print(f'kernel time: {time.time() - ts}s')
    sys.exit(0)

    program = Program(img_h, img_w, init_method='growth')
    eval_program_fitness_gpu(program, img_h, img_w, data_size, dataset, stack, conv_buffer, hist_buffer, label)

    ts = time.time()
    for _ in range(500):
        program = Program(img_h, img_w, init_method='growth')
        eval_program_fitness_gpu(program, img_h, img_w, data_size, dataset, stack, conv_buffer, hist_buffer, label)
        # print(program.fitness)
    print(f'kernel time: {time.time() - ts}s')


    stack_res = stack.copy_to_host()

    stack_res = stack_res.reshape(img_h * img_w, -1)
    for i in range(0, img_h * img_w):
        for j in range(0, min(data_size, 10)):
            print(stack_res[i, j], end=' ')
        print()
