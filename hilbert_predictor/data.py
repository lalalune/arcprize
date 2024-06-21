import numpy as np
import os
import json
from .gilbert2d import flatten_2d_to_1d

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
                
                
        # train_data.append(rules_input)
        # test_data.append(test_input)
        train_data.append(rules_input_hilbert)
        test_data.append(test_input_hilbert)
    return train_data, test_data

# Load training data
training_data_dir = "./data/training"
evaluating_data_dir = "./data/evaluation"

# get all files in training_data_dir that end with .json
training_file_paths = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir) if f.endswith('.json')]
evaluating_file_paths = [os.path.join(evaluating_data_dir, f) for f in os.listdir(evaluating_data_dir) if f.endswith('.json')]

training_train_data, training_test_data = load_data(training_file_paths)
evaluating_train_data, evaluating_test_data = load_data(evaluating_file_paths)

def pad_examples(examples, max_length=1024):
    padded_examples = []
    for example in examples:
        input_example, output_example = example
        input_padded = np.pad(input_example, (max_length - len(input_example), 0), 'constant', constant_values=10)
        output_padded = np.pad(output_example, (0, max_length - len(output_example)), 'constant', constant_values=10)
        padded_examples.append((input_padded, output_padded))
    return padded_examples

padded_train_data = [pad_examples(data) for data in training_train_data]
padded_test_data = [pad_examples(data) for data in training_test_data]

def save_to_file(data, file_path):
    with open(file_path, 'w') as f:
        for examples in data:
            for example in examples:
                input_example, output_example = example
                assert len(input_example) == 1024 and len(output_example) == 1024, "Input and output should each be 1024 elements"
                f.write(' '.join(map(str, input_example)) + ' ' + ' '.join(map(str, output_example)) + '\n')
                

save_to_file(padded_train_data, 'padded_train_data.txt')
save_to_file(padded_test_data, 'padded_test_data.txt')


## Split file here

padded_test_data = []
padded_train_data = []

with open('padded_test_data.txt', 'r') as f:
    for line in f:
        tokens = line.strip().split(' ')
        for i in range(len(tokens)):
            tokens[i] = int(tokens[i])
        input_example = np.array(tokens, dtype=np.int64)
        padded_test_data.append(input_example)
        
with open('padded_train_data.txt', 'r') as f:
    for line in f:
        tokens = line.strip().split(' ')
        # for each token, read as an int
        for i in range(len(tokens)):
            tokens[i] = int(tokens[i])
        input_example = np.array(tokens, dtype=np.int64)
        padded_train_data.append(input_example)

print("Training data loaded")
print('training_train_data:', len(training_train_data))
print('training_test_data:', len(training_test_data))

# print the first training example
print(training_train_data[0])

print("Padded train data:")
# print padded_train_data
print(padded_train_data)
# get padded_train_data shape
print("Padded train data shape:")
for i, data in enumerate(padded_train_data):
    print(f"Shape of padded_train_data[{i}]: {np.array(data).shape}")
# get the dimensions of the padded_train_data
print(np.array(padded_train_data).shape)