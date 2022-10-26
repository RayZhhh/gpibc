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


@cuda.jit()
def cu_region(dataset, stack, img_h, img_w):
    # the pixel coordinates that this thread responsible for
    px = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    if px < img_h and py < img_w:
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = _pixel_value_in_dataset(dataset, px, py, img_h, img_w)
    cuda.syncthreads()


@cuda.jit()
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
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum
    cuda.syncthreads()


@cuda.jit()
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
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum / 16
    cuda.syncthreads()


@cuda.jit()
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
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum
    cuda.syncthreads()


@cuda.jit()
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
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum
    cuda.syncthreads()


@cuda.jit()
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
        stack[_pixel_ind_in_stack(px, py, img_h, img_w)] = sum
    cuda.syncthreads()


@cuda.jit()
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


@cuda.jit()
def cu_hist_eq_statistic(stack, hist_buffer, img_h, img_w, rx, ry, rh, rw):
    """Args launching or invoking this function:
    Grid dim: [image_number]
    Block dim: [1]
    """
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


@cuda.jit()
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