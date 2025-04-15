import numpy as np
from scipy.ndimage import label, find_objects, center_of_mass, rotate
from collections import Counter
import itertools
import copy # For deep copying grids during transformations

# --- 1. Grid Representation & Basic Utilities ---
# Grids are represented as NumPy arrays.

def display_grid(grid):
    """Simple text-based display for a grid."""
    for row in grid:
        print(" ".join(map(str, row)))
    print("-" * len(grid[0]) * 2)

# --- 2. Perception Module ---

def find_grid_objects(grid, connectivity=1):
    """
    Identifies connected components (objects) of the same color, ignoring color 0 (black/background).
    Returns:
        labeled_grid: Grid where each object has a unique integer ID.
        num_objects: Total number of objects found.
        object_slices: List of slice objects to bounding box each object.
        object_colors: Dictionary mapping object ID to its color.
        object_masks: Dictionary mapping object ID to its boolean mask.
        object_sizes: Dictionary mapping object ID to its pixel count.
        object_centers: Dictionary mapping object ID to its center of mass (row, col).
    """
    background_color = 0
    labeled_grid = np.zeros_like(grid, dtype=int)
    object_colors = {}
    object_masks = {}
    object_sizes = {}
    object_centers = {}
    current_label = 0

    # Iterate through unique colors present (excluding background)
    unique_colors = np.unique(grid[grid != background_color])
    for color in unique_colors:
        color_mask = (grid == color)
        labeled_mask, num_features = label(color_mask, structure=np.ones((3,3)) if connectivity==2 else [[0,1,0],[1,1,1],[0,1,0]]) # Use 4 or 8 connectivity
        
        # Renumber labels to be unique across colors
        new_labels = labeled_mask + current_label * (labeled_mask > 0)
        labeled_grid = np.maximum(labeled_grid, new_labels) # Combine results
        
        for i in range(1, num_features + 1):
            obj_id = current_label + i
            mask = (new_labels == obj_id)
            object_colors[obj_id] = color
            object_masks[obj_id] = mask
            object_sizes[obj_id] = np.sum(mask)
            center = center_of_mass(mask) # Returns (row, col)
            object_centers[obj_id] = center
            
        current_label += num_features
        
    num_objects = current_label
    # find_objects needs the labeled array without gaps in numbering if possible
    # Recalculate slices based on the final combined labeled_grid
    object_slices = find_objects(labeled_grid) 
    
    # Adjust object_slices to match the obj_ids we assigned (find_objects gives N slices for N labels)
    # This part needs careful mapping if labels are not sequential from 1.
    # For simplicity, let's assume find_objects gives slices corresponding to obj_id 1, 2, ... N
    # A more robust implementation would map slices correctly based on label values.
    
    return labeled_grid, num_objects, object_slices, object_colors, object_masks, object_sizes, object_centers


def get_global_properties(grid):
    """Calculates global properties like color counts, symmetry."""
    properties = {}
    colors, counts = np.unique(grid, return_counts=True)
    properties['color_counts'] = dict(zip(colors, counts))
    
    # Basic Symmetry Checks (can be expanded)
    properties['is_symmetric_vertical'] = np.array_equal(grid, np.fliplr(grid))
    properties['is_symmetric_horizontal'] = np.array_equal(grid, np.flipud(grid))
    # Diagonal symmetry is more complex to check perfectly for non-square grids
    
    return properties

# --- 3. Domain-Specific Language (DSL) Operations ---
# These functions take a grid and parameters, returning a *new* modified grid.
# They might need object information from the perception module.

def op_move_object(grid, object_masks, obj_id, dr, dc):
    """Moves a specific object by (dr, dc). Returns modified grid."""
    if obj_id not in object_masks:
        return grid # Object not found
        
    mask = object_masks[obj_id]
    original_color = grid[mask][0] # Assumes uniform color
    new_grid = grid.copy()
    new_grid[mask] = 0 # Erase original object
    
    rows, cols = np.where(mask)
    new_rows, new_cols = rows + dr, cols + dc
    
    # Check bounds
    valid = (new_rows >= 0) & (new_rows < grid.shape[0]) & \
            (new_cols >= 0) & (new_cols < grid.shape[1])
            
    new_grid[new_rows[valid], new_cols[valid]] = original_color
    return new_grid

def op_recolor_object(grid, object_masks, obj_id, new_color):
    """Recolors a specific object."""
    if obj_id not in object_masks:
        return grid
    
    new_grid = grid.copy()
    new_grid[object_masks[obj_id]] = new_color
    return new_grid
    
def op_fill_rect(grid, top, left, bottom, right, color):
    """Fills a rectangular area."""
    new_grid = grid.copy()
    new_grid[top:bottom+1, left:right+1] = color
    return new_grid

def op_rotate_object(grid, object_masks, obj_id, angle, center):
    """ Rotates an object around a center point. COMPLEX to implement perfectly with grid pixels"""
    # NOTE: Pixel-perfect grid rotation is non-trivial. 
    # scipy.ndimage.rotate can rotate arrays, but aligning perfectly with object boundaries/centers 
    # and handling pixel interpolation/discretization is complex.
    # This is a placeholder for a potentially complex operation.
    print(f"Warning: op_rotate_object is complex and not fully implemented here.")
    # A simple approach might extract the object bounding box, rotate that subgrid, 
    # then place it back, handling overlaps.
    return grid 
    
# --- Add more operations: delete, copy, grow, fill_holes, draw_line, etc. ---
# Each operation needs careful implementation to handle grid boundaries, object definitions etc.


# --- 4. Rule Representation & Hypothesis ---
# A rule could be represented as a sequence of operations.
# Rule Hypothesis Generation is the core challenge. 
# This example uses a *very* basic approach: try a fixed set of simple rules.

class Rule:
    def __init__(self, description, operations):
        self.description = description
        # Operations: list of tuples (function, params_dict)
        # Params_dict might include specific values OR references to perceived properties
        # e.g., {'obj_id': 'largest', 'dr': 1, 'dc': 0}
        self.operations = operations 

    def apply(self, input_grid):
        """Applies the sequence of operations to an input grid."""
        current_grid = copy.deepcopy(input_grid)
        
        # --- Perception Step within Apply ---
        # Re-run perception on the *current* grid state if needed by operations
        # Or, pass initial perception results if operations only refer to initial state.
        labeled_grid, num_obj, _, _, obj_masks, obj_sizes, obj_centers = find_grid_objects(current_grid)

        for op_func, params in self.operations:
            # --- Parameter Resolution ---
            # Resolve symbolic params (like 'largest') using perception results
            resolved_params = {}
            obj_id_to_use = None

            for key, val in params.items():
                if key == 'obj_id' and isinstance(val, str): # Symbolic object reference
                    if val == 'largest' and obj_sizes:
                         obj_id_to_use = max(obj_sizes, key=obj_sizes.get)
                    elif val == 'smallest' and obj_sizes:
                        obj_id_to_use = min(obj_sizes, key=obj_sizes.get)
                    # Add more symbolic refs: 'topmost', 'red_one', etc. Needs more perception.
                    else:
                        print(f"Warning: Unknown symbolic object reference '{val}'")
                        # Fallback or error
                        
                    if obj_id_to_use:
                         resolved_params['obj_id'] = obj_id_to_use
                    else:
                         # Cannot apply this rule if symbolic ref fails
                         print(f"Could not resolve symbolic reference {val}. Skipping operation.")
                         return input_grid # Return original grid on failure? Or raise error?
                         
                # Resolve other symbolic parameters (e.g., color based on frequency)
                #elif key == 'new_color' and val == 'most_frequent': ...
                
                else:
                    resolved_params[key] = val

            # Add necessary perception results to params if needed by the op function
            if 'object_masks' not in resolved_params:
                 resolved_params['object_masks'] = obj_masks
            # Add other perception data as required by ops...

            # --- Execute Operation ---
            try:
                 current_grid = op_func(current_grid, **resolved_params)
                 # Update perception info if subsequent operations depend on the new state
                 if any(p in ['obj_id', 'object_masks'] for p in itertools.chain.from_iterable(op[1].keys() for op in self.operations[self.operations.index((op_func, params))+1:])):
                     labeled_grid, num_obj, _, _, obj_masks, obj_sizes, obj_centers = find_grid_objects(current_grid)
                     # Update the masks available for the *next* step
                     
            except Exception as e:
                 print(f"Error applying operation {op_func.__name__} with params {resolved_params}: {e}")
                 return input_grid # Failed to apply rule

        return current_grid

    def __str__(self):
        return self.description


def generate_hypotheses(input_grid, output_grid):
    """
    *Simplified* Hypothesis Generation. Returns a list of candidate Rule objects.
    A real system needs program synthesis here.
    """
    hypotheses = []
    
    # Example simple hypotheses:
    # 1. Move the largest object 1 step right
    hypotheses.append(Rule("Move largest object +1 col", 
                          [(op_move_object, {'obj_id': 'largest', 'dr': 0, 'dc': 1})]))
    # 2. Recolor the smallest object to blue (color 1)
    hypotheses.append(Rule("Recolor smallest object to blue",
                          [(op_recolor_object, {'obj_id': 'smallest', 'new_color': 1})]))
    # 3. Move all objects down by 2 steps (requires iterating or a dedicated op)
    # ... add more predefined simple rules or combinations ...
    
    # --- Placeholder for real synthesis ---
    # A real system would:
    # - Analyze input/output differences using perception.
    # - Generate candidate DSL programs systematically.
    # - Prune search space based on complexity and partial matches.
    print("--- Using simplified hypothesis generation ---")

    return hypotheses


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
    Takes a specific rule that worked and tries to abstract it.
    e.g., replace 'blue' with 'color C' if color varied but role was same.
    This is highly complex and task-dependent. Placeholder for now.
    """
    print("--- Abstraction step (conceptual) ---")
    # For now, just return the concrete rule found
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
        print(f"\nGenerated {len(candidate_rules)} candidate rules.")
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
        """
        if not self.generalized_rule:
            print("No rule has been learned. Please call fit() first.")
            return [copy.deepcopy(ex['input']) for ex in test_examples] if test_examples else []
        predictions = []
        for i, test_ex in enumerate(test_examples):
            test_input_grid = test_ex['input']
            print(f"\nTest Input {i+1}:")
            display_grid(test_input_grid)
            predicted_test_output = self.generalized_rule.apply(test_input_grid)
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