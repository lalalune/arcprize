import numpy as np
import os
import json
from .gilbert2d import flatten_2d_to_1d

PAD_TOKEN = 10
START_TOKEN = 11
END_TOKEN = 12

def load_and_process_data(file_paths, max_context_length=8192):
    processed_data = []
    
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        train_examples = data['train']
        test_examples = data['test']
        
        # Create combinations of train examples (1 to 4) for each test example
        for test_example in test_examples:
            for num_train_examples in range(1, min(5, len(train_examples) + 1)):
                context = []
                for train_example in train_examples[:num_train_examples]:
                    context.extend(flatten_2d_to_1d(np.array(train_example['input'])))
                    context.extend(flatten_2d_to_1d(np.array(train_example['output'])))
                    context.append(END_TOKEN)
                
                # Add test input
                context.extend(flatten_2d_to_1d(np.array(test_example['input'])))
                context.append(START_TOKEN)
                
                # Pad or truncate context to max_context_length
                if len(context) < max_context_length:
                    context = [PAD_TOKEN] * (max_context_length - len(context)) + context
                else:
                    context = context[-max_context_length:]
                
                # Prepare target (test output)
                target = flatten_2d_to_1d(np.array(test_example['output']))
                target = np.pad(target, (0, max_context_length - len(target)), 'constant', constant_values=PAD_TOKEN)
                
                processed_data.append((np.array(context), target))
    
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