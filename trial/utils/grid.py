

import numpy as np


def generate_grid(search_size, score_size, central=False):
    stride = search_size[0] / score_size[0]
    shift_x = (np.arange(0, score_size[1], dtype=np.float32) + 0.5) * stride
    shift_y = (np.arange(0, score_size[0], dtype=np.float32) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    grid = np.concatenate((shift_x[..., None], shift_y[..., None]), axis=-1)
    if central:
        center = np.array(search_size, dtype=np.float32).reshape((1, 1, -1)) / 2.
        grid = grid - center
    return grid


# if __name__ == '__main__':
#     a = generate_grid([128, 256], [16, 32], central=True)
#     pass
