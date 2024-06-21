import numpy as np
import os
import json
from .gilbert2d import flatten_2d_to_1d

PAD_TOKEN = 10
START_EXAMPLE_TOKEN = 11
END_EXAMPLE_TOKEN = 12
START_SEQUENCE_TOKEN = 13
END_SEQUENCE_TOKEN = 14
NUM_TOKENS = 15

def load_and_process_data(file_paths, max_context_length=8192):
    processed_data = []
    unpadded_strings = []
    
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        train_examples = data['train']
        test_examples = data['test']
        
        for test_example in test_examples:
            for num_train_examples in range(1, min(5, len(train_examples) + 1)):
                context = [START_SEQUENCE_TOKEN]
                unpadded_context = [START_SEQUENCE_TOKEN]
                
                for train_example in train_examples[:num_train_examples]:
                    context.append(START_EXAMPLE_TOKEN)
                    unpadded_context.append(START_EXAMPLE_TOKEN)
                    
                    train_input = flatten_2d_to_1d(np.array(train_example['input']))
                    train_output = flatten_2d_to_1d(np.array(train_example['output']))
                    context.extend(train_input)
                    context.extend(train_output)
                    unpadded_context.extend(train_input)
                    unpadded_context.extend(train_output)
                    
                    context.append(END_EXAMPLE_TOKEN)
                    unpadded_context.append(END_EXAMPLE_TOKEN)
                
                context.append(START_EXAMPLE_TOKEN)
                unpadded_context.append(START_EXAMPLE_TOKEN)
                test_input = flatten_2d_to_1d(np.array(test_example['input']))
                context.extend(test_input)
                unpadded_context.extend(test_input)
                context.append(END_EXAMPLE_TOKEN)
                unpadded_context.append(END_EXAMPLE_TOKEN)
                
                context.append(END_SEQUENCE_TOKEN)
                unpadded_context.append(END_SEQUENCE_TOKEN)
                
                # Pad the context if necessary
                if len(context) < max_context_length:
                    context = context + [PAD_TOKEN] * (max_context_length - len(context))
                else:
                    context = context[:max_context_length]
                
                target = [START_SEQUENCE_TOKEN] + flatten_2d_to_1d(np.array(test_example['output'])) + [END_SEQUENCE_TOKEN]
                target = target + [PAD_TOKEN] * (max_context_length - len(target))
                
                processed_data.append((np.array(context), np.array(target)))
                unpadded_strings.append(' '.join(map(str, unpadded_context)))
    
    with open('hilbert_data.txt', 'w') as f:
        for string in unpadded_strings:
            f.write(string + '\n')
    
    return processed_data

# Load and process data
training_data_dir = "./data/training"
evaluating_data_dir = "./data/evaluation"

training_file_paths = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir) if f.endswith('.json')]
evaluating_file_paths = [os.path.join(evaluating_data_dir, f) for f in os.listdir(evaluating_data_dir) if f.endswith('.json')]

training_data = load_and_process_data(training_file_paths)
evaluating_data = load_and_process_data(evaluating_file_paths)

# Save processed data
np.save('processed_training_data.npy', training_data)
np.save('processed_evaluating_data.npy', evaluating_data)