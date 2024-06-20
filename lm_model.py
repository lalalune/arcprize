import os
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import requests
import torch

OPENAI_API_KEY = ""
CLAUDE_API_KEY = ""

# check if keys.txt exists
if os.path.exists("keys.txt"):
    # read keys.txt
    with open("keys.txt", "r") as f:
        lines = f.readlines()
        OPENAI_API_KEY = lines[0].strip().replace("OPENAI_API_KEY=", "")
        CLAUDE_API_KEY = lines[1].strip().replace("CLAUDE_API_KEY=", "")
elif os.environ.get("OPENAI_API_KEY") is not None and os.environ.get("CLAUDE_API_KEY") is not None:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
else:
    print("Please add your OpenAI API key and Claude API key to a file called keys.txt or set them as environment variables")
    exit()

def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay)) # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by)) # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield(x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield(x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2)
        yield from generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                              -bx2, -by2, -(ax-ax2), -(ay-ay2))
        
def flatten_2d_to_1d(array_2d):
    height, width = len(array_2d), len(array_2d[0])
    array_1d = [None] * (width * height)
    
    for idx, (x, y) in enumerate(gilbert2d(width, height)):
        array_1d[idx] = array_2d[y][x]
    
    return array_1d


def unflatten_1d_to_2d(array_1d, width, height):
    array_2d = [[None] * width for _ in range(height)]
    
    for idx, (x, y) in enumerate(gilbert2d(width, height)):
        array_2d[y][x] = array_1d[idx]
    
    return array_2d

# Load the data from the data/train directory containing the json files
training_data_dir = "./data/training"
evaluating_data_dir = "./data/evaluation"

# get all files in training_data_dir that end with .json
training_file_paths = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir) if f.endswith('.json')]

evaluating_file_paths = [os.path.join(evaluating_data_dir, f) for f in os.listdir(evaluating_data_dir) if f.endswith('.json')]

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

_float_colors = [tuple(c / 255 for c in col) for col in colors_rgb.values()]
arc_cmap = ListedColormap(_float_colors)

class ArcColors:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    FUCHSIA = 6
    ORANGE = 7
    TEAL = 8
    BROWN = 9

def plot_grid(grid1: np.ndarray, grid2: np.ndarray = None):
    if grid2 is None:
        fig, ax = plt.subplots()
        ax.pcolormesh(
            grid1,
            cmap=arc_cmap,
            rasterized=True,
            vmin=0,
            vmax=9,
        )
        ax.set_xticks(np.arange(0, grid1.shape[1], 1))
        ax.set_yticks(np.arange(0, grid1.shape[0], 1))
        ax.grid()
        ax.set_aspect(1)
        ax.invert_yaxis()
        plt.show()
        return
    
    fig, axs = plt.subplots(1, 2)

    axs[0].pcolormesh(
        grid1,
        cmap=arc_cmap,
        rasterized=True,
        vmin=0,
        vmax=9,
    )
    axs[0].set_xticks(np.arange(0, grid1.shape[1], 1))
    axs[0].set_yticks(np.arange(0, grid1.shape[0], 1))
    axs[0].grid()
    axs[0].set_aspect(1)
    axs[0].invert_yaxis()

    axs[1].pcolormesh(
        grid2,
        cmap=arc_cmap,
        rasterized=True,
        vmin=0,
        vmax=9,
    )
    axs[1].set_xticks(np.arange(0, grid2.shape[1], 1))
    axs[1].set_yticks(np.arange(0, grid2.shape[0], 1))
    axs[1].grid()
    axs[1].set_aspect(1)
    axs[1].invert_yaxis()
    plt.show()

def load_data(file_paths):
    train_data = []
    test_data = []
    for file_path in file_paths:
        rules_input = []
        rules_input_hilbert = []
        test_input = []
        test_input_hilbert = []
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data['train']:
                input_hilbert = flatten_2d_to_1d(np.array(item['input']))
                output_hilbert = flatten_2d_to_1d(np.array(item['output']))
                                
                rules_input.append([
                    np.array(item['input'], dtype=np.int64),
                    np.array(item['output'], dtype=np.int64)
                ])
                
                rules_input_hilbert.append([
                    np.array(input_hilbert, dtype=np.int64),
                    np.array(output_hilbert, dtype=np.int64)
                ])
                
            for item in data['test']:
                test_input.append([
                    np.array(item['input'], dtype=np.int64),
                    np.array(item['output'], dtype=np.int64)
                ])
                
                input_hilbert = flatten_2d_to_1d(np.array(item['input']))
                output_hilbert = flatten_2d_to_1d(np.array(item['output']))
                
                test_input_hilbert.append([
                    np.array(input_hilbert, dtype=np.int64),
                    np.array(output_hilbert, dtype=np.int64)
                ])
                
                
        train_data.append(rules_input)
        test_data.append(test_input)
    return train_data, test_data

training_train_data, training_test_data = load_data(training_file_paths)
evaluating_train_data, evaluating_test_data = load_data(evaluating_file_paths)

def normalize_data(data):
    return data / 9.0

def denormalize_data(data):
    return data * 9.0

def expand_squared_matrix(matrix, size):
    max_size = size
    current_size = matrix.shape[0]
    if current_size == max_size:
        return normalize_data(matrix)
    
    ratio = max_size / current_size
    floor_ratio = int(np.floor(ratio))
    
    if floor_ratio * current_size == max_size:
        return normalize_data(matrix.repeat(floor_ratio, axis=0).repeat(floor_ratio, axis=1))
    
    resized_matrix = matrix.repeat(floor_ratio, axis=0).repeat(floor_ratio, axis=1)
    pad_size = max_size - resized_matrix.shape[0]
    
    padded_matrix = np.full((max_size, max_size), 0)
    padded_matrix[pad_size//2:pad_size//2+resized_matrix.shape[0], pad_size//2:pad_size//2+resized_matrix.shape[1]] = resized_matrix
    
    return normalize_data(padded_matrix)

def expand_rectangular_matrix(matrix, size):
    max_size = 32
    current_size = matrix.shape[0]
    current_width = matrix.shape[1]
    if current_size == max_size and current_width == max_size:
        return normalize_data(matrix)
    ratio = max_size // current_size
    ratio_width = max_size // current_width
    divisible = ratio * current_size == max_size and ratio_width * current_width == max_size
    if divisible is True:
       return normalize_data(matrix.repeat(ratio, axis=0).repeat(ratio_width, axis=1))

    ## if the size is not divisible by 32
    ## we need to add padding and center the reiszed image
    floor_ratio = np.floor(ratio)
    floor_ratio_width = np.floor(ratio_width)
    resized_matrix = matrix.repeat(floor_ratio, axis=0).repeat(floor_ratio_width, axis=1)
    pad_size = max_size - resized_matrix.shape[0]
    pad_size_width = max_size - resized_matrix.shape[1]
    padded_matrix = np.full(size, 0)
    padded_matrix[pad_size//2:pad_size//2+resized_matrix.shape[0], pad_size_width//2:pad_size_width//2+resized_matrix.shape[1]] = resized_matrix
    return normalize_data(padded_matrix)

def expand_matrix(matrix, size):
    if matrix.shape[0] == matrix.shape[1]:
        return expand_squared_matrix(matrix, size[0])
    return expand_rectangular_matrix(matrix, size)

MAX_SHAPE = (30, 30)

def extract_data_for_transformer(train, test, max_shape=MAX_SHAPE):
    pairs = []
    for i, element in enumerate(train):
        expanded_0 = expand_matrix(element[0], max_shape)
        expanded_1 = expand_matrix(element[1], max_shape)
        pairs.append(expanded_0)
        pairs.append(expanded_1)

    final_test = expand_matrix(test[0][0], max_shape)
    attended_output = expand_matrix(test[0][1], max_shape)
    return pairs, final_test, attended_output

def extract_batch(data, test):
    batches = []
    for i in range(0, len(data)):
        extracted = extract_data_for_transformer(data[i], test[i])
        batches.append(extracted)
        ### augment the data
        for j in range(3):
            rotated = [[np.rot90(x, j+1) for x in extracted[0]], np.rot90(extracted[1], j+1), np.rot90(extracted[2], j+1)]
            batches.append(rotated)
            batches.append([[np.flip(x, 0) for x in rotated[0]], np.flip(rotated[1], 0), np.flip(rotated[2], 0)])
        batches.append([[np.flip(x, 1) for x in extracted[0]], np.flip(extracted[1], 1), np.flip(extracted[2], 1)])
    return batches

# This is what the data looks like, although the input and output can have different dimensions (and different from each other)
# {"train": [{"input": [[3, 3, 8], [3, 7, 0], [5, 0, 0]], "output": [[0, 0, 5], [0, 7, 3], [8, 3, 3]]}, {"input": [[5, 5, 2], [1, 0, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 0, 1], [2, 5, 5]]}], "test": [{"input": [[6, 3, 5], [6, 8, 0], [4, 0, 0]], "output": [[0, 0, 4], [0, 8, 6], [5, 3, 6]]}]}

# for each input output in the training data, we need to flatten the input and output
examples = []
for item in training_train_data[0]:
    input_hilbert = flatten_2d_to_1d(np.array(item[0]))
    output_hilbert = flatten_2d_to_1d(np.array(item[1]))
    
    # instead of hilbert flattening, just reshape row wise into a 1d
    # input_hilbert = np.array(item[0]).reshape(-1)
    # output_hilbert = np.array(item[1]).reshape(-1)
    
    examples.append([
        np.array(input_hilbert, dtype=np.int64),
        np.array(output_hilbert, dtype=np.int64)
    ])
    
# now get the test input
test_input = []
for item in training_test_data[0]:
    input_hilbert = flatten_2d_to_1d(np.array(item[0]))
    output_hilbert = flatten_2d_to_1d(np.array(item[1]))
    # input_hilbert = np.array(item[0]).reshape(-1)
    # output_hilbert = np.array(item[1]).reshape(-1)    
    
    test_input.append([
        np.array(input_hilbert, dtype=np.int64),
        np.array(output_hilbert, dtype=np.int64)
    ])
    
prompt = "Determine the output for the input that has no output mapping (=>). Your answer should be formatted as a list of symbols. Only give the output, no commentary.\n"

mapping = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J'
}

for example in examples:
    # make sure example[0] is a list of numbers with commas between them
    prompt += f"{' '.join(map(str, example[0]))} => {' '.join(map(str, example[1]))}\n"

# for test, just give the input
for item in test_input:
    prompt += f"{' '.join(map(str, example[0]))} =>"

# for each key as a string in prompt, replace with the mapping value instead
for key, value in mapping.items():
    prompt = prompt.replace(str(key), value)
    
# get the arg from openai api key
    
# send to openai chatgpt
response = requests.post("https://api.openai.com/v1/chat/completions", json={
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ]
}, headers={
    "Authorization": "Bearer " + OPENAI_API_KEY,
    "Content-Type": "application/json",
})

response = response.json()

print(response)

openai_output = response['choices'][0]['message']['content']

# for each key as a string in prompt, replace with the mapping value instead
for key, value in mapping.items():
    openai_output = openai_output.replace(value, str(key))

API_URL = "https://api.anthropic.com/v1/messages"

headers = {
    "x-api-key": CLAUDE_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}
data = {
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 4096,
    "messages": [
        {"role": "user", "content": prompt}
    ]
}

response = requests.post(API_URL, headers=headers, data=json.dumps(data))
response_data = response.json()
output = response_data["content"][0]["text"]

# for each key as a string in prompt, replace with the mapping value instead
for key, value in mapping.items():
    output = output.replace(value, str(key))

print (prompt)

print("output")
print(output)

print("openai_output")
print(openai_output)

# find json``` and remove json``` and everything before
output = output[output.find("```"):]
# find ``` and remove everything after
output = output[:output.find("```")]
print("actual")
# print the answer
for item in test_input:
    print(item[1])
