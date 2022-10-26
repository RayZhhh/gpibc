# This file defines device side functions and fit evaluation kernels,
# which allows evaluating multiple programs simultaneously.


import math
from numba import cuda
import fset

MAX_PIXEL_VALUE = 255
MAX_PROGRAM_LEN = 200
MAX_TOP = 10


# -The device side dataset is structured as follows:
#                [0 0 0]  [1 1 1]  [2 2 2]                   [0 0 0 0 0 0 0 0 0]
#     raw image: [0 0 0]  [1 1 1]  [2 2 2] ... => reshape => [1 1 1 1 1 1 1 1 1] => transpose =>
#                [0 0 0]  [1 1 1]  [2 2 2]                   [2 2 2 2 2 2 2 2 2]
#
#              [<---- data_size ---->]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#     dataset: [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]


@cuda.jit(device=True, inline=True)
def _dataset_value(dataset, data_size, im_w, i, j):
    pixel_row = i * im_w + j
    pixel_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return dataset[pixel_row * data_size + pixel_col]


# -Also a buffer for storing temp conv value is allocated, which is structured as follows:
#   The shape of stack is the same as the conv buffer.
#                  [<---- data_size ---->]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#     program 0 => [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [=====================]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#     program 1 => [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [.....................]


@cuda.jit(device=True, inline=True)
def _pixel_index_in_stack(data_size, im_h, im_w, i, j) -> int:
    program_no = cuda.blockIdx.y
    pixel_row = i * im_w + j
    pixel_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return program_no * data_size * im_h * im_w + pixel_row * data_size + pixel_col


@cuda.jit(device=True, inline=True)
def _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j):
    return stack[_pixel_index_in_stack(data_size, im_h, im_w, i, j)]


@cuda.jit(device=True, inline=True)
def _pixel_conv_buffer_index(data_size, im_h, im_w, i, j) -> int:
    program_no = cuda.blockIdx.y
    pixel_row = i * im_w + j
    pixel_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return program_no * data_size * im_h * im_w + pixel_row * data_size + pixel_col


@cuda.jit(device=True, inline=True)
def _pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j):
    return buffer[_pixel_conv_buffer_index(data_size, im_h, im_w, i, j)]


# -The buffer to store the std value is structured as follows:
#     top=0 => [000000000000000000000]
#              [111111111111111111111]
#              [222222222222222222222]
#              [-------MAX TOP-------]
#     top=1 => [000000000000000000000]
#              [111111111111111111111]
#              [222222222222222222222]
#              [-------MAX TOP-------]
#              [.....................]


@cuda.jit(device=True, inline=True)
def _std_res_index(top, data_size) -> int:
    program_no = cuda.blockIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return program_no * MAX_TOP * data_size + top * data_size + col


@cuda.jit(device=True, inline=True)
def _std_res_value(std_res, top, data_size):
    return std_res[_std_res_index(top, data_size)]


@cuda.jit(device=True)
def g_std(stack, data_size, im_h, im_w, rx, ry, rh, rw, std_res, top):
    """Calculate the standard deviation of the region."""
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        avg = 0
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                avg += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j)
        avg /= rh * rw
        deviation = 0
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                value = _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) - avg
                deviation += value * value
        deviation /= rh * rw
        deviation = math.sqrt(deviation)
        std_res[_std_res_index(top, data_size)] = deviation
    cuda.syncthreads()


@cuda.jit(device=True)
def sub(std_res, top, data_size):
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        res1 = _std_res_value(std_res, top - 2, data_size)
        res2 = _std_res_value(std_res, top - 1, data_size)
        std_res[_std_res_index(top - 2, data_size)] = res1 - res2
    cuda.syncthreads()


@cuda.jit(device=True)
def region(dataset, data_size, im_h, im_w, stack):
    """Both Region_S and Region_R execute this function,
    the only difference is the region size which is stored in the local memory.
    This function copy the image from dataset to stack.
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(im_h):
            for j in range(im_w):
                d = _dataset_value(dataset, data_size, im_w, i, j)
                stack[_pixel_index_in_stack(data_size, im_h, im_w, i, j)] = d
    cuda.syncthreads()


@cuda.jit(device=True)
def lap(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
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
        # calculate each pixel result
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                sum -= _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 4
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j)
                sum = max(0, sum)
                sum = min(MAX_PIXEL_VALUE, sum)
                buffer[_pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = _pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = _pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)
    cuda.syncthreads()


@cuda.jit(device=True)
def gau1(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Achieve parallel Gau1 function
    In this implementation, a thread is responsible for an image.
    After Gau1 operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Gaussian smooth kernel is: [1, 2, 1]
                                   [2, 4, 2] * (1 / 16).
                                   [1, 2, 1]
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 2
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 2
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 4
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 2
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 2
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1)
                sum /= 16
                sum = max(0, sum)
                sum = min(MAX_PIXEL_VALUE, sum)
                buffer[_pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = _pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = _pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)
    cuda.syncthreads()


@cuda.jit(device=True)
def sobel_x(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Perform Sobel_X on image.
    In this implementation, a thread is responsible for an image.
    After Sobel_X operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Sobel Vertical kernel is: [ 1, 2, 1]
                                  [ 0, 0, 0]
                                  [-1,-2,-1].
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 2
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1)
                sum -= _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1)
                sum -= _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 2
                sum -= _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1)
                sum = max(0, sum)
                sum = min(MAX_PIXEL_VALUE, sum)
                buffer[_pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = _pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = _pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)
    cuda.syncthreads()


@cuda.jit(device=True)
def sobel_y(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Perform Sobel_Y on image.
    In this implementation, a thread is responsible for an image.
    After Sobel_Y operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Sobel Horizontal kernel is: [-1, 0, 1 ]
                                    [-2, 0, 2 ]
                                    [-1, 0, 1 ].
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                sum -= _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1)
                sum -= _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 2
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 2
                sum -= _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1)
                sum = max(0, sum)
                sum = min(MAX_PIXEL_VALUE, sum)
                buffer[_pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = _pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = _pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)
    cuda.syncthreads()


@cuda.jit(device=True)
def log1(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Perform LoG1 on image.
        In this implementation, a thread is responsible for an image.
        After LoG1 operation, rx += 2; ry += 2; rh -= 4; rw -= 4.

        The LoG kernel is: [0,  0,  1,  0,  0]
                           [0,  1,  2,  1,  0]
                           [1,  2,-16,  2,  1]
                           [0,  1,  2,  1,  0]
                           [0,  0,  1,  0,  0].
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 2, rx + rh - 2):
            for j in range(ry + 2, ry + rw - 2):
                sum = 0
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 2
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 2)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 2) * 2
                sum -= _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 16
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 2
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 2)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 2
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1)
                sum += _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j)
                sum = max(0, sum)
                sum = min(MAX_PIXEL_VALUE, sum)
                buffer[_pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 2, rx + rh - 2):
            for j in range(ry + 2, ry + rw - 2):
                stack_index = _pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = _pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)
    cuda.syncthreads()


@cuda.jit(device=True)
def lbp(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Perform Local Binary Pattern operation to images.
    Step 1:
        calculate the value of each pixel based on the threshold
        pixel_lbp(i) = 0 if pixel(i) < center else 1

    Step 2:
        calculate the value of the center pixel using the weights: [  1,  2,  4]
                                                                   [128,  C,  8]
                                                                   [ 64, 32, 16]
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                cp_value = _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j)
                p_1 = _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1)
                p_2 = _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j)
                p_4 = _pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1)
                p_8 = _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1)
                p_16 = _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1)
                p_32 = _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j)
                p_64 = _pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1)
                p_128 = _pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1)
                sum = 0
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
                buffer[_pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = _pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = _pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)
    cuda.syncthreads()


# -A buffer for Histogram based functions is allocated, which is structured as follows:
#                  [<---- data_size ---->]
#                  [0 0 0 0 0 0 0 0 0 0 .]
#                  [1 1 1 1 1 1 1 1 1 1 .]
#                  [2 2 2 2 2 2 2 2 2 2 .]
#                  [3 3 3 3 3 3 3 3 3 3 .]
#     program 0 => [4 4 4 4 4 4 4 4 4 4 .]
#                  [5 5 5 5 5 5 5 5 5 5 .]
#                  [.....................]
#                  [---------256---------]
#                  [0 0 0 0 0 0 0 0 0 0 .]
#                  [1 1 1 1 1 1 1 1 1 1 .]
#                  [2 2 2 2 2 2 2 2 2 2 .]
#                  [3 3 3 3 3 3 3 3 3 3 .]
#     program 0 => [4 4 4 4 4 4 4 4 4 4 .]
#                  [5 5 5 5 5 5 5 5 5 5 .]
#                  [.....................]
#                  [---------256---------]


@cuda.jit(device=True, inline=True)
def _hist_buffer_index(data_size, value) -> int:
    program_no = cuda.blockIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return program_no * (MAX_PIXEL_VALUE + 1) * data_size + value * data_size + col


@cuda.jit(device=True, inline=True)
def _hist_buffer_value(buffer, data_size, value):
    return buffer[_hist_buffer_index(data_size, value)]


@cuda.jit(device=True)
def hist_eq(stack, data_size, im_h, im_w, rx, ry, rh, rw, hist_buffer):
    """Performing Historical Equalisation to the input region."""
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        # clear the buffer
        for i in range(0, MAX_PIXEL_VALUE + 1):
            hist_buffer[_hist_buffer_index(data_size, i)] = 0

        # statistic intensity of each pixel, this process can not perform coalesced access
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                pixel_value = int(_pixel_value_in_stack(stack, data_size, im_h, im_w, i, j))
                pixel_value = max(0, pixel_value)
                pixel_value = min(MAX_PIXEL_VALUE, pixel_value)
                hist_buffer[_hist_buffer_index(data_size, pixel_value)] += 1

        # uniform
        pixel_num = rh * rw
        for i in range(0, MAX_PIXEL_VALUE + 1):
            hist_buffer[_hist_buffer_index(data_size, i)] /= pixel_num

        # add up
        for i in range(1, MAX_PIXEL_VALUE + 1):
            hist_buffer[_hist_buffer_index(data_size, i)] += \
                hist_buffer[_hist_buffer_index(data_size, i - 1)]

        # mapping table
        for i in range(0, MAX_PIXEL_VALUE + 1):
            hist_buffer[_hist_buffer_index(data_size, i)] *= (MAX_PIXEL_VALUE - 1)

        # update the intensity value of each pixel of the region
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                raw = int(_pixel_value_in_stack(stack, data_size, im_h, im_w, i, j))
                raw = max(0, raw)
                raw = min(MAX_PIXEL_VALUE, raw)
                new_value = _hist_buffer_value(hist_buffer, data_size, raw)
                stack[_pixel_index_in_stack(data_size, im_h, im_w, i, j)] = new_value
    cuda.syncthreads()


# -This kernel allows evaluating multiple programs in the same time.
# Grid dims when launching this kernel: (int((DATA_SIZE - 1 + THREAD_PER_BLOCK) / THREAD_PER_BLOCK), POP_SIZE_TO_EVAL)
# Block dims when launching this kernel: THREAD_PER_BLOCK


@cuda.jit()
def calc_pop_fit(name, rx, ry, rh, rw, plen, img_h, img_w, data_size, dataset, stack, conv_buffer, hist_buffer,
                 std_res):
    # the program that the thread is responsible for
    program_no = cuda.blockIdx.y

    # top which point to the std_res
    top = 0
    reg_x, reg_y, reg_h, reg_w = 0, 0, 0, 0

    # reverse iteration
    for i in range(plen[program_no] - 1, -1, -1):
        if name[program_no][i] == fset.Region_R:
            reg_x, reg_y, reg_h, reg_w = rx[program_no][i], ry[program_no][i], rh[program_no][i], rw[program_no][i]
            region(dataset, data_size, img_h, img_w, stack)

        elif name[program_no][i] == fset.Region_S:
            reg_x, reg_y, reg_h, reg_w = rx[program_no][i], ry[program_no][i], rh[program_no][i], rw[program_no][i]
            region(dataset, data_size, img_h, img_w, stack)

        elif name[program_no][i] == fset.G_Std:
            g_std(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, std_res, top)
            top += 1

        elif name[program_no][i] == fset.Hist_Eq:
            hist_eq(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, hist_buffer)

        elif name[program_no][i] == fset.Gau1:
            gau1(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == fset.Gau11:
            pass

        elif name[program_no][i] == fset.GauXY:
            pass

        elif name[program_no][i] == fset.Lap:
            lap(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == fset.Sobel_X:
            sobel_x(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == fset.Sobel_Y:
            sobel_y(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == fset.LoG1:
            log1(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 2, reg_y + 2, reg_h - 4, reg_w - 4

        elif name[program_no][i] == fset.LoG2:
            pass

        elif name[program_no][i] == fset.HOG:
            pass

        elif name[program_no][i] == fset.Sub:
            sub(std_res, top, data_size)
            top -= 1

        else:
            print('Error: Do not support the function')

    if top != 1:
        print('error: top != 1')
