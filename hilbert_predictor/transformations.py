import numpy as np


def identity(x):
    return x


identity.is_identity = True


def flip_horizontal(matrix):
    return np.fliplr(matrix)


flip_horizontal.is_identity = False


def flip_vertical(matrix):
    return np.flipud(matrix)


flip_vertical.is_identity = False


def rotate_180(matrix):
    return np.rot90(matrix, 2)


rotate_180.is_identity = False