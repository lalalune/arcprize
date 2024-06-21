"""
An example of using the wave function collapse with 2D image.

"""
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from .wfc import WaveFunctionCollapse

color_black = (0x00, 0x00, 0x00)
colors_rgb_noblack = {
    1: (0x00, 0x74, 0xD9),
    2: (0xFF, 0x41, 0x36),
    3: (0x2E, 0xCC, 0x40),
    4: (0xFF, 0xDC, 0x00),
    5: (0xA0, 0xA0, 0xA0),
    6: (0xF0, 0x12, 0xBE),
    7: (0xFF, 0x85, 0x1B),
    8: (0x7F, 0xDB, 0xFF),
    9: (0x87, 0x0C, 0x25),
}

colors_rgb = {
    0: (0x00, 0x00, 0x00),
    1: (0x00, 0x74, 0xD9),
    2: (0xFF, 0x41, 0x36),
    3: (0x2E, 0xCC, 0x40),
    4: (0xFF, 0xDC, 0x00),
    5: (0xA0, 0xA0, 0xA0),
    6: (0xF0, 0x12, 0xBE),
    7: (0xFF, 0x85, 0x1B),
    8: (0x7F, 0xDB, 0xFF),
    9: (0x87, 0x0C, 0x25),
}

def plot_patterns(patterns, title=''):
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title, fontsize=16)
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        if i > len(patterns):
            break
        fig.add_subplot(rows, columns, i)
        show(patterns[i - 1])

    # plt.show()


def load_sample(path):
    sample = plt.imread(path)
    # Expand dim to 3D
    sample = np.expand_dims(sample, axis=0)
    sample = sample[:, :, :, :3]

    return sample


def show(image):
    if image.shape[0] == 1:
        return plt.imshow(np.squeeze(image, axis=0))


if __name__ == '__main__':

    grid_size = (1, 10, 10)
    pattern_size = (1, 2, 2)

    pattern_obj_size = (4, 8)
    # randomly get 1-4 colors from the color palette and create a pattern
    number_of_colors = np.random.randint(1, 4)
    colors = np.random.choice(list(colors_rgb_noblack.keys()), size=number_of_colors)
    # randomly add 0-5 black colors to the palette
    number_of_black_colors = np.random.randint(0, 5)
    colors = np.concatenate((colors, np.zeros(number_of_black_colors)))
    # if the length of colors is still 1, add a black color
    if len(colors) == 1:
        colors = np.concatenate((colors, np.zeros(1)))
    # fill a pattern with random colors
    pattern = np.random.choice(colors, size=pattern_obj_size)
    
    # create a bitmap image using the actual values of the pattern
    pattern_image = np.zeros((pattern_obj_size[0], pattern_obj_size[1], 3), dtype=np.uint8)
    for i in range(pattern_obj_size[0]):
        for j in range(pattern_obj_size[1]):
            pattern_image[i, j] = colors_rgb[pattern[i, j]]

    # Convert the pattern_image to a PIL Image object
    pil_image = Image.fromarray(pattern_image)

    # Save the PIL Image as a PNG file
    pil_image.save('temp.png')
    

    sample = load_sample('temp.png')
    show(sample)

    # plt.show()

    wfc = WaveFunctionCollapse(grid_size, sample, pattern_size)
    # plot_patterns(wfc.get_patterns(), 'patterns')

    # _, _, legal_patterns = wfc.propagator.legal_patterns(wfc.patterns[2], (0, 0, 1))
    # show(Pattern.from_index(2).to_image())
    # plt.show()
    # plot_patterns([Pattern.from_index(i).to_image() for i in legal_patterns])

    fig, ax = plt.subplots()
    image = wfc.get_image()
    im = show(image)
    while True:
        done = wfc.step()
        if done:
            break
        image = wfc.get_image()

        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
            im.set_array(image)

        fig.canvas.draw()
        plt.pause(0.001)

    plt.show()
