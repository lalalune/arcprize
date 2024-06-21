from termcolor import colored
import math
from .gilbert2d import *
import numpy as np
from .data import PAD_TOKEN


def unflatten_1d_to_2d_viz(array_1d, width=None, height=None):
    print(array_1d.shape)
    print(array_1d.shape[0] ** 0.5)
    val1 = int(math.ceil(array_1d.shape[0] ** 0.5))
    height = val1
    width = val1
    expected_length = width * height

    if len(array_1d) < expected_length:
        array_1d = np.pad(
            array_1d, (0, expected_length - len(array_1d)), constant_values=PAD_TOKEN
        )

    array_2d = [[None] * width for _ in range(height)]
    for idx, (x, y) in enumerate(gilbert2d(width, height)):
        if idx < len(array_1d):
            array_2d[y][x] = array_1d[idx]

    return array_2d


from termcolor import colored


def color_string(array):
    color_map = {
        0: "black",
        1: "green",
        2: "yellow",
        3: "blue",
        4: "magenta",
        5: "cyan",
        6: "white",
        7: "light_red",
        8: "red",
        9: "green",
    }
    colored_string = "".join(
        colored(f"{elem:2}", color_map.get(elem, "white")) + " " for elem in array
    )
    return colored_string
