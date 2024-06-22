import json
import numpy as np
import os
from random import randint
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from .wfc import WaveFunctionCollapse  # Ensure this is properly implemented

# Color palette
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

# make a custom cmap
cmap = plt.cm.colors.ListedColormap([np.array(c) / 255 for c in colors_rgb.values()])


def generate_random_pattern(size, max_colors=3):
    # Ensure that a variety of colors is chosen, including at least one non-black color if possible
    colors = np.random.choice(
        range(1, 10), size=max_colors, replace=False
    )  # Choose from non-black colors
    colors = np.insert(colors, 0, 0)  # Ensure black is also included

    # Create the pattern using chosen colors
    pattern = np.random.choice(colors, size=size)

    # Add random shapes with non-black colors
    for _ in range(2):  # Adding two random shapes
        shape_color = np.random.choice(colors[1:])  # Choose non-black colors for shapes
        shape_size = (randint(1, size[0] // 2), randint(1, size[1] // 2))
        shape_pos = (
            randint(0, size[0] - shape_size[0]),
            randint(0, size[1] - shape_size[1]),
        )
        pattern[
            shape_pos[0] : shape_pos[0] + shape_size[0],
            shape_pos[1] : shape_pos[1] + shape_size[1],
        ] = shape_color

    return pattern.astype(
        np.uint8
    )  # Ensure the pattern is in uint8 for image processing


def save_pattern_as_image(pattern, filename):
    if pattern.dtype != np.uint8:
        # Assume pattern is normalized between 0 and 1 for floating point
        pattern = (255 * pattern).astype(np.uint8)
    img = Image.fromarray(pattern)
    img.save(filename)


def wfc_pattern_expansion(input_pattern, output_size, start_point):
    # Convert input pattern indices to RGB values for the image
    pattern_image = np.zeros(
        (input_pattern.shape[0], input_pattern.shape[1], 3), dtype=np.uint8
    )
    for i in range(input_pattern.shape[0]):
        for j in range(input_pattern.shape[1]):
            pattern_image[i, j] = colors_rgb[input_pattern[i, j]]

    # Save the pattern image as 'temp.png' for the WFC module
    pil_image = Image.fromarray(pattern_image)
    pil_image.save("temp.png")

    # Read 'temp.png' back into an array for WFC processing
    sample = plt.imread("temp.png")
    sample = np.expand_dims(
        sample, axis=0
    )  # Convert to a batch of images with one image
    sample = sample[:, :, :, :3]  # Remove any alpha channel

    # Initialize WFC with the sample and specified output size
    wfc = WaveFunctionCollapse((1, *output_size), sample, (1, 2, 2))
    while True:
        done = wfc.step()
        if done:
            break

    output_pattern_rgb = np.squeeze(wfc.get_image(), axis=0)

    # Convert output pattern RGB values back to indices based on the input color palette
    output_pattern = np.zeros(output_size, dtype=np.uint8)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            rgb_value = tuple(output_pattern_rgb[i, j] * 255)  # Convert to 0-255 range
            min_distance = float("inf")
            closest_color_index = None
            for color_index, color_rgb in colors_rgb.items():
                distance = sum(abs(c1 - c2) for c1, c2 in zip(rgb_value, color_rgb))
                if distance < min_distance:
                    min_distance = distance
                    closest_color_index = color_index
            output_pattern[i, j] = closest_color_index

    return output_pattern


def save_plots(input_pattern, output_pattern, filename, title="Pattern"):
    """Save the plots of input and output patterns side by side to a file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    ax1.imshow(input_pattern, interpolation="nearest", cmap=cmap)
    ax1.set_title("Input")
    ax1.axis("off")
    ax2.imshow(output_pattern, interpolation="nearest", cmap=cmap)
    ax2.set_title("Output")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def generate_and_save_plots(input_pattern, output_pattern, index, output_dir):
    """Generate and save plots for input and output patterns side by side."""
    plot_filename = os.path.join(output_dir, f"example_{index}_plot.png")
    save_plots(input_pattern, output_pattern, plot_filename, title=f"Example {index}")
    return plot_filename


def generate_challenge_examples(
    input_pattern, output_pattern, num_examples, output_dir
):
    examples = []
    for i in range(num_examples):
        # Flip input and output patterns horizontally and/or vertically
        flip_h = np.random.choice([True, False])
        flip_v = np.random.choice([True, False])
        input_flipped = np.flip(
            input_pattern, axis=(0 if flip_v else 1) if flip_h else ()
        )
        output_flipped = np.flip(
            output_pattern, axis=(0 if flip_v else 1) if flip_h else ()
        )

        # Change colors randomly
        color_map = np.random.permutation(np.arange(len(colors_rgb)))
        input_color_changed = color_map[input_flipped]
        output_color_changed = color_map[output_flipped]

        plot_filename = generate_and_save_plots(
            input_color_changed, output_color_changed, i, output_dir
        )

        example = {
            "input": input_color_changed.tolist(),
            "output": output_color_changed.tolist(),
            "plot": plot_filename,
        }
        examples.append(example)
    return examples


def generate_fewshot_challenges(num_challenges, train_ratio, output_dir):
    challenges = []
    num_train_challenges = int(num_challenges * train_ratio)
    num_eval_challenges = num_challenges - num_train_challenges

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, "training")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    for i in range(num_train_challenges):
        input_size = (randint(3, 6), randint(3, 6))
        output_size = (randint(6, 16), randint(6, 16))
        num_examples_per_challenge = randint(2, 4)

        input_pattern = generate_random_pattern(input_size, randint(2, 4))
        output_pattern = wfc_pattern_expansion(
            input_pattern, output_size, (0, 0)
        )  # example start_point

        examples = generate_challenge_examples(
            input_pattern, output_pattern, num_examples_per_challenge, train_dir
        )
        challenge = {
            "train": examples,
            "test": [
                {"input": input_pattern.tolist(), "output": output_pattern.tolist()}
            ],
        }

        challenge_hash = f"{randint(0, 0xFFFFFFFF):08x}"
        with open(os.path.join(train_dir, f"{challenge_hash}.json"), "w") as f:
            json.dump(challenge, f, indent=4)

    for i in range(num_eval_challenges):
        input_size = (randint(2, 6), randint(2, 6))
        output_size = (randint(input_size[0], 10), randint(input_size[1], 10))
        num_examples_per_challenge = randint(2, 4)

        input_pattern = generate_random_pattern(input_size, randint(2, 4))
        output_pattern = wfc_pattern_expansion(
            input_pattern, output_size, (0, 0)
        )  # example start_point

        examples = generate_challenge_examples(
            input_pattern, output_pattern, num_examples_per_challenge, eval_dir
        )
        challenge = {
            "train": examples,
            "test": [
                {"input": input_pattern.tolist(), "output": output_pattern.tolist()}
            ],
        }

        challenge_hash = f"{randint(0, 0xFFFFFFFF):08x}"
        with open(os.path.join(eval_dir, f"{challenge_hash}.json"), "w") as f:
            json.dump(challenge, f, indent=4)


if __name__ == "__main__":
    num_challenges = 1000
    train_ratio = 0.95
    output_dir = "fewshot_challenges"
    generate_fewshot_challenges(num_challenges, train_ratio, output_dir)
    print(f"Fewshot challenges generated and saved in {output_dir}")
