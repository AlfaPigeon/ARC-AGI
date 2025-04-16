import json
import os
import random

# Directory to save the generated training data
OUTPUT_DIR = os.path.join('data', 'preschool', 'training')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Directory to save the generated evaluation data
EVAL_OUTPUT_DIR = os.path.join('data', 'preschool', 'evaluation')
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

def fill_row_rule(size, row_idx, value):
    mat = [[0 for _ in range(size)] for _ in range(size)]
    for j in range(size):
        mat[row_idx][j] = value
    return mat

def fill_col_rule(size, col_idx, value):
    mat = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        mat[i][col_idx] = value
    return mat

def invert_rule(mat):
    return [[0 if cell else 1 for cell in row] for row in mat]

def diagonal_rule(size, value):
    mat = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        mat[i][i] = value
    return mat

def generate_basic_tasks():
    tasks = []
    size_options = [2, 3]
    # Rule 1: Fill a row with 1s
    for size in size_options:
        for row in range(size):
            task = {
                'filename': f'fill_row_{size}_{row}.json',
                'train': [
                    {'input': fill_row_rule(size, row, 1), 'output': fill_row_rule(size, row, 1)},
                    {'input': fill_row_rule(size, (row+1)%size, 1), 'output': fill_row_rule(size, (row+1)%size, 1)}
                ],
                'test': [
                    {'input': fill_row_rule(size, (row+2)%size, 1), 'output': fill_row_rule(size, (row+2)%size, 1)}
                ]
            }
            tasks.append(task)
    # Rule 2: Fill a column with 1s
    for size in size_options:
        for col in range(size):
            task = {
                'filename': f'fill_col_{size}_{col}.json',
                'train': [
                    {'input': fill_col_rule(size, col, 1), 'output': fill_col_rule(size, col, 1)},
                    {'input': fill_col_rule(size, (col+1)%size, 1), 'output': fill_col_rule(size, (col+1)%size, 1)}
                ],
                'test': [
                    {'input': fill_col_rule(size, (col+2)%size, 1), 'output': fill_col_rule(size, (col+2)%size, 1)}
                ]
            }
            tasks.append(task)
    # Rule 3: Invert matrix
    for size in size_options:
        mat1 = [[random.randint(0,1) for _ in range(size)] for _ in range(size)]
        mat2 = [[random.randint(0,1) for _ in range(size)] for _ in range(size)]
        task = {
            'filename': f'invert_{size}.json',
            'train': [
                {'input': mat1, 'output': invert_rule(mat1)},
                {'input': mat2, 'output': invert_rule(mat2)}
            ],
            'test': [
                {'input': invert_rule(mat1), 'output': mat1},
                {'input': invert_rule(mat2), 'output': mat2}
            ]
        }
        tasks.append(task)
    # Rule 4: Diagonal fill
    for size in size_options:
        task = {
            'filename': f'diagonal_{size}.json',
            'train': [
                {'input': diagonal_rule(size, 1), 'output': diagonal_rule(size, 1)},
                {'input': diagonal_rule(size, 0), 'output': diagonal_rule(size, 0)}
            ],
            'test': [
                {'input': diagonal_rule(size, 1), 'output': diagonal_rule(size, 1)}
            ]
        }
        tasks.append(task)
    return tasks

def generate_simple_matrices():
    tasks = generate_basic_tasks()
    for task in tasks:
        # Write to training directory
        train_path = os.path.join(OUTPUT_DIR, task['filename'])
        with open(train_path, 'w') as f:
            json.dump({'train': task['train'], 'test': task['test']}, f, indent=2)
        print(f"Wrote {train_path}")
        # Write to evaluation directory
        eval_path = os.path.join(EVAL_OUTPUT_DIR, task['filename'])
        with open(eval_path, 'w') as f:
            json.dump({'train': task['train'], 'test': task['test']}, f, indent=2)
        print(f"Wrote {eval_path}")

if __name__ == "__main__":
    generate_simple_matrices()
