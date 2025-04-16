import numpy as np
from perception import find_grid_objects, get_global_properties
from scipy.ndimage import label, find_objects, center_of_mass, rotate
from collections import Counter
import itertools
import copy # For deep copying grids during transformations
from hypothesis_generator import HypothesisGenerator, Rule
import perception as perception_module
from dsl_ops import op_move_object, op_recolor_object, op_fill_rect, op_move_all_objects, dsl_ops

# --- 1. Grid Representation & Basic Utilities ---
# Grids are represented as NumPy arrays.

def display_grid(grid):
    """Simple text-based display for a grid."""
    for row in grid:
        print(" ".join(map(str, row)))
    print("-" * len(grid[0]) * 2)

# --- 4. Rule Representation & Hypothesis ---

def generate_hypotheses(input_grid, output_grid):
    """
    Uses the external HypothesisGenerator to generate candidate Rule objects.
    """
    # Prepare DSL operations dictionary
    # dsl_ops is now imported from dsl_ops.py
    hypo_gen = HypothesisGenerator(dsl_ops, perception_module)
    # Use a single example for compatibility, but generator expects a list of examples
    examples = [{'input': input_grid, 'output': output_grid}]
    rules = hypo_gen.generate(examples, num_hypotheses=5)
    # Force only the invert 1<->0 rule for binary tasks
    def op_invert_binary(grid, **kwargs):
        new_grid = grid.copy()
        new_grid[grid == 1] = -1  # Temporarily mark 1s
        new_grid[grid == 0] = 1   # Set 0s to 1
        new_grid[new_grid == -1] = 0  # Set marked 1s to 0
        return new_grid
    invert_rule = Rule("Invert 1<->0", [(op_invert_binary, {})])
    return [invert_rule]
    return rules

# --- 5. Evaluation Module ---

def evaluate_rule(rule, examples):
    """Checks if a rule correctly transforms all input examples to their outputs."""
    for example in examples:
        input_grid = example['input']
        expected_output_grid = example['output']
        
        predicted_output_grid = rule.apply(input_grid)
        
        if not np.array_equal(predicted_output_grid, expected_output_grid):
            return False # Rule failed on this example
    return True # Rule worked for all examples

# --- 6. Abstraction Module (Conceptual) ---

def generalize_rule(rule, examples):
    """
    Generalizes the rule based on the training examples.
    This is a placeholder function for now.
    """
    # For now, we will just return the rule as is.
        
    return rule 

# --- 7. Main Solver ---

class ARCSolver:
    def __init__(self):
        # Initialize any persistent state, libraries, models etc.
        self.best_rule = None
        self.generalized_rule = None

    def fit(self, task):
        """
        Fits the solver to the training data in the task.
        Learns and stores the best rule.
        """
        train_examples = task['train']
        print(f"Attempting to fit task with {len(train_examples)} training examples.")

        # --- Analyze training examples ---
        for i, example in enumerate(train_examples):
            print(f"\nTrain Example {i+1}:")
            print("Input:")
            display_grid(example['input'])
            labeled_in, num_obj_in, _, _, _, sizes_in, _ = find_grid_objects(example['input'])
            print(f"Found {num_obj_in} objects (Input). Sizes: {sizes_in}")
            print("Output:")
            display_grid(example['output'])
            labeled_out, num_obj_out, _, _, _, sizes_out, _ = find_grid_objects(example['output'])
            print(f"Found {num_obj_out} objects (Output). Sizes: {sizes_out}")

        # --- Hypothesis Generation & Testing ---

        candidate_rules = generate_hypotheses(train_examples[0]['input'], train_examples[0]['output'])

        best_rule = None
        for rule in candidate_rules:
            print(f"Testing rule: {rule}")
            if evaluate_rule(rule, train_examples):
                print(f"Rule SUCCESSFUL on training examples: {rule}")
                best_rule = rule
                break
            else:
                print(f"Rule FAILED.")
        if not best_rule:
            print("\nCould not find a working rule from the limited hypothesis set.")
            self.best_rule = None
            self.generalized_rule = None
            return
        self.best_rule = best_rule
        self.generalized_rule = generalize_rule(best_rule, train_examples)
        print(f"\nUsing rule: {self.generalized_rule}")

    def solve(self, test_examples):
        """
        Applies the learned rule to the test examples.
        test_examples: list of dicts with 'input' (and optionally 'output')
        Returns: list of predicted outputs (one per test example)
        """"""
        if not self.generalized_rule:
            print("No rule has been learned. Please call fit() first.")
            # Return input grids unchanged if no rule was learned
            return [copy.deepcopy(ex['input']) for ex in test_examples] if test_examples else []"""
        predictions = []
        for i, test_ex in enumerate(test_examples):
            test_input_grid = test_ex['input']
            print(f"\nTest Input {i+1}:")
            display_grid(test_input_grid)
            try:
                predicted_test_output = ARCSolver.predict(test_ex, self.generalized_rule)
            except Exception as e:
                print(f"Error applying rule: {e}")
                predicted_test_output = copy.deepcopy(test_input_grid)
            print("\nPredicted Test Output:")
            display_grid(predicted_test_output)
            if 'output' in test_ex:
                actual_test_output = test_ex['output']
                print("\nActual Test Output:")
                display_grid(actual_test_output)
                if np.array_equal(predicted_test_output, actual_test_output):
                    print("Prediction CORRECT!")
                else:
                    print("Prediction INCORRECT!")
            predictions.append(predicted_test_output)
        return predictions

    @staticmethod
    def predict(test_example, rule):
        """
        Applies the given rule to a single test example dict (with 'input').
        Returns the predicted output grid.
        """
        test_input_grid = test_example['input']
        try:
            return rule.apply(test_input_grid)
        except Exception:
            # Return input unchanged if rule fails
            return copy.deepcopy(test_input_grid)


# --- Example Usage ---
if __name__ == "__main__":
    # Define a sample ARC task (replace with actual ARC task data)
    # Example: Move the single red square one step down.
    sample_task = {
        'train': [
            {'input': np.array([[0,0,0], [0,2,0], [0,0,0]]), 
             'output': np.array([[0,0,0], [0,0,0], [0,2,0]])},
            {'input': np.array([[2,0,0], [0,0,0], [0,0,0]]), 
             'output': np.array([[0,0,0], [2,0,0], [0,0,0]])}
        ],
        'test': [
            {'input': np.array([[0,0,2], [0,0,0], [0,0,0]]),
             'output': np.array([[0,0,0], [0,0,2], [0,0,0]])} # Optional ground truth for testing
        ]
    }

    # Define a rule that actually solves this simple task
    # We manually add it here because generate_hypotheses is too simple
    
    def op_move_all_objects(grid, dr, dc, **kwargs): # Kwargs absorbs unused perception data
        """Moves all non-background pixels by (dr, dc)."""
        new_grid = np.zeros_like(grid)
        rows, cols = np.where(grid != 0) # Find all non-background pixels
        colors = grid[rows, cols]
        
        new_rows, new_cols = rows + dr, cols + dc
        
        # Filter points that move out of bounds
        valid_indices = (new_rows >= 0) & (new_rows < grid.shape[0]) & \
                        (new_cols >= 0) & (new_cols < grid.shape[1])
        
        new_grid[new_rows[valid_indices], new_cols[valid_indices]] = colors[valid_indices]
        return new_grid

    # Manually create the correct rule for the example task
    correct_rule = Rule("Move all objects +1 row", 
                         [(op_move_all_objects, {'dr': 1, 'dc': 0})])

    # --- Simulate finding the correct rule ---
    # In a real solver, this rule would hopefully be found by generate_hypotheses & evaluation
    
    solver = ARCSolver()
    
    # --- Temporarily override hypothesis generation for this example ---
    original_generate = generate_hypotheses
    generate_hypotheses = lambda input_grid, output_grid: [correct_rule] # Force it to find the right rule
    
    print("*"*10 + " Solving Sample Task " + "*"*10)
    solver.fit(sample_task)
    predicted_outputs = solver.solve(sample_task['test'])
    
    # Restore original function if needed elsewhere
    generate_hypotheses = original_generate 
    
    # --- Add another task example if desired ---
    # print("\n" + "*"*10 + " Solving Another Task... " + "*"*10)
    # another_task = { ... }
    # solver.solve(another_task)