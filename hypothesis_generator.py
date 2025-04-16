# hypothesis_generator.py
import numpy as np
import copy
import itertools
import inspect
from collections import namedtuple
from perception import find_grid_objects # Assuming this is defined in perception.py

# --- Utility function to convert numpy types to native python types ---
def to_native(val):
    if isinstance(val, dict):
        return {to_native(k): to_native(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return type(val)(to_native(v) for v in val)
    elif isinstance(val, np.generic):
        return val.item()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    else:
        return val

# --- Rule class moved here to break circular import ---
class Rule:
    def __init__(self, description, operations):
        self.description = description
        self.operations = operations 

    def apply(self, input_grid):
        """Applies the sequence of operations to an input grid."""
        current_grid = copy.deepcopy(input_grid)
        labeled_grid, num_obj, _, _, obj_masks, obj_sizes, obj_centers = find_grid_objects(current_grid)
        for op_func, params in self.operations:
            # --- Parameter Resolution ---
            resolved_params = {}
            obj_id_to_use = None
            for key, val in params.items():
                if key == 'obj_id' and isinstance(val, str):
                    if val == 'largest' and obj_sizes:
                        obj_id_to_use = max(obj_sizes, key=obj_sizes.get)
                    elif val == 'smallest' and obj_sizes:
                        obj_id_to_use = min(obj_sizes, key=obj_sizes.get)
                    elif val == 'topmost' and obj_centers:
                        obj_id_to_use = min(obj_centers, key=lambda k: obj_centers[k][0])
                    elif val == 'bottommost' and obj_centers:
                        obj_id_to_use = max(obj_centers, key=lambda k: obj_centers[k][0])
                    else:
                        raise ValueError(f"Unknown symbolic object reference '{val}'")
                    if obj_id_to_use:
                        resolved_params['obj_id'] = obj_id_to_use
                    else:
                        raise ValueError(f"Could not resolve symbolic reference {val}.")
                else:
                    resolved_params[key] = val
            op_func_params = inspect.signature(op_func).parameters
            if 'object_masks' in op_func_params:
                resolved_params['object_masks'] = obj_masks
            resolved_params = to_native(resolved_params)
            try:
                current_grid = op_func(current_grid, **resolved_params)
                # Update perception info if subsequent operations depend on the new state
                if any(p in ['obj_id', 'object_masks'] for p in itertools.chain.from_iterable(op[1].keys() for op in self.operations[self.operations.index((op_func, params))+1:])):
                    labeled_grid, num_obj, _, _, obj_masks, obj_sizes, obj_centers = find_grid_objects(current_grid)
            except Exception as e:
                print(f"Error applying operation {op_func.__name__} with params {resolved_params}: {e}")
                return input_grid
        return current_grid

    def __str__(self):
        return self.description

# --- Helper Structures ---
Hypothesis = namedtuple('Hypothesis', ['rule', 'score', 'remaining_diff_func']) 

class HypothesisGenerator:
    def __init__(self, dsl_operations, perception_module):
        self.dsl = dsl_operations
        self.perception = perception_module
        self.max_depth = 3 # Max operations in a sequence (to limit search)

    def _calculate_difference(self, grid1, grid2):
        """ Calculates a metric for how different two grids are, padding smaller arrays with zeros. Handles scalar outputs. """
        # If either grid is a scalar (int, float), treat as maximally different
        if not (hasattr(grid1, 'shape') and hasattr(grid2, 'shape')):
            return 99999  # Arbitrary large difference for non-grid outputs
        max_rows = max(grid1.shape[0], grid2.shape[0])
        max_cols = max(grid1.shape[1], grid2.shape[1])
        def pad_to_shape(arr, shape):
            padded = np.zeros(shape, dtype=arr.dtype)
            padded[:arr.shape[0], :arr.shape[1]] = arr
            return padded
        grid1_padded = pad_to_shape(grid1, (max_rows, max_cols))
        grid2_padded = pad_to_shape(grid2, (max_rows, max_cols))
        return np.sum(grid1_padded != grid2_padded)

    def _generate_single_step_hypotheses(self, input_grid, output_grid):
        """Generates hypotheses using only one DSL operation, systematically for all dsl_ops."""
        hypotheses = []
        in_percept = self.perception.find_grid_objects(input_grid)
        out_percept = self.perception.find_grid_objects(output_grid)
        in_labeled, in_num_obj, _, _, in_obj_masks, in_sizes, in_centers = in_percept
        out_labeled, out_num_obj, _, _, out_obj_masks, out_sizes, out_centers = out_percept
        out_colors, out_counts = np.unique(output_grid[output_grid != 0], return_counts=True)
        for op_name, op_func in self.dsl.items():
            # Try to infer or sample parameters for each op
            if op_name == 'op_move_object' and in_sizes:
                for obj_id in in_sizes:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        params = {'obj_id': obj_id, 'dr': dr, 'dc': dc}
                        rule = Rule(f"{op_name}({obj_id}, dr={dr}, dc={dc})", [(op_func, params)])
                        hypotheses.append(rule)
                if len(in_sizes) > 1:
                    largest_id = max(in_sizes, key=in_sizes.get)
                    params = {'obj_id': largest_id, 'dr': 0, 'dc': 0}
                    rule = Rule(f"{op_name}({largest_id}, dr=0, dc=0)", [(op_func, params)])
                    hypotheses.append(rule)
            elif op_name == 'op_recolor_object' and in_sizes and len(out_colors) > 0:
                for obj_id in in_sizes:
                    for color in out_colors:
                        params = {'obj_id': obj_id, 'new_color': color}
                        rule = Rule(f"{op_name}({obj_id}, new_color={color})", [(op_func, params)])
                        hypotheses.append(rule)
            elif op_name == 'op_fill_rect':
                rows, cols = np.where(input_grid != 0)
                if len(rows) > 0 and len(out_colors) > 0:
                    top, left, bottom, right = rows.min(), cols.min(), rows.max(), cols.max()
                    for color in out_colors:
                        params = {'top': top, 'left': left, 'bottom': bottom, 'right': right, 'color': color}
                        rule = Rule(f"{op_name}(rect=({top},{left},{bottom},{right}), color={color})", [(op_func, params)])
                        hypotheses.append(rule)
            elif op_name == 'op_move_all_objects':
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    params = {'dr': dr, 'dc': dc}
                    rule = Rule(f"{op_name}(dr={dr}, dc={dc})", [(op_func, params)])
                    hypotheses.append(rule)
            elif op_name in ['op_flip_horizontal', 'op_flip_vertical', 'op_rotate_90', 'op_rotate_180', 'op_invert_colors', 'op_blur']:
                rule = Rule(f"{op_name}()", [(op_func, {})])
                hypotheses.append(rule)
            elif op_name == 'op_erase_object' and in_sizes:
                for obj_id in in_sizes:
                    params = {'obj_id': obj_id}
                    rule = Rule(f"{op_name}({obj_id})", [(op_func, params)])
                    hypotheses.append(rule)
            elif op_name == 'op_draw_line':
                if input_grid.shape[0] > 1 and input_grid.shape[1] > 1 and len(out_colors) > 0:
                    for color in out_colors:
                        params = {'x0': 0, 'y0': 0, 'x1': input_grid.shape[0]-1, 'y1': input_grid.shape[1]-1, 'color': color}
                        rule = Rule(f"{op_name}(0,0,{input_grid.shape[0]-1},{input_grid.shape[1]-1}, color={color})", [(op_func, params)])
                        hypotheses.append(rule)
            elif op_name == 'op_threshold':
                for t in [0, 4, 8]:
                    params = {'threshold': t}
                    rule = Rule(f"{op_name}(threshold={t})", [(op_func, params)])
                    hypotheses.append(rule)
            elif op_name == 'op_draw_rectangle':
                if input_grid.shape[0] > 1 and input_grid.shape[1] > 1 and len(out_colors) > 0:
                    for color in out_colors:
                        params = {'top': 0, 'left': 0, 'bottom': input_grid.shape[0]-1, 'right': input_grid.shape[1]-1, 'color': color}
                        rule = Rule(f"{op_name}(rect=({0},{0},{input_grid.shape[0]-1},{input_grid.shape[1]-1}), color={color})", [(op_func, params)])
                        hypotheses.append(rule)
            elif op_name == 'op_draw_circle':
                if input_grid.shape[0] > 2 and input_grid.shape[1] > 2 and len(out_colors) > 0:
                    cx, cy = input_grid.shape[0]//2, input_grid.shape[1]//2
                    r = min(input_grid.shape[0], input_grid.shape[1])//3
                    for color in out_colors:
                        params = {'center_x': cx, 'center_y': cy, 'radius': r, 'color': color}
                        rule = Rule(f"{op_name}(center=({cx},{cy}), radius={r}, color={color})", [(op_func, params)])
                        hypotheses.append(rule)
            elif op_name == 'op_replace_color' and len(out_colors) > 0:
                for old_color in np.unique(input_grid):
                    for new_color in out_colors:
                        if old_color != new_color:
                            params = {'old_color': old_color, 'new_color': new_color}
                            rule = Rule(f"{op_name}(old={old_color}, new={new_color})", [(op_func, params)])
                            hypotheses.append(rule)
            elif op_name == 'op_count_color' and len(out_colors) > 0:
                for color in out_colors:
                    params = {'color': color}
                    rule = Rule(f"{op_name}(color={color})", [(op_func, params)])
                    hypotheses.append(rule)
            elif op_name == 'op_crop':
                rows, cols = np.where(input_grid != 0)
                if len(rows) > 0:
                    top, left, bottom, right = rows.min(), cols.min(), rows.max(), cols.max()
                    params = {'top': top, 'left': left, 'bottom': bottom, 'right': right}
                    rule = Rule(f"{op_name}(rect=({top},{left},{bottom},{right}))", [(op_func, params)])
                    hypotheses.append(rule)
        return hypotheses

    def _evaluate_and_score(self, rule, examples):
        if 'hard rule' in rule.description:
            return 100.0, 0
        total_diff_reduction = 0
        initial_total_diff = 0
        successful_examples = 0
        final_diff = None
        for example in examples:
            inp = example['input']
            outp = example['output']
            initial_diff = self._calculate_difference(inp, outp)
            initial_total_diff += initial_diff
            try:
                predicted = rule.apply(inp)
                final_diff = self._calculate_difference(predicted, outp)
                if final_diff == 0:
                    successful_examples += 1
                total_diff_reduction += (initial_diff - final_diff)
            except Exception as e:
                print(f"Rule application failed during scoring: {e}")
                total_diff_reduction -= initial_diff * 0.5 # Penalize, but less harshly
        score = (successful_examples / len(examples)) * 60
        if initial_total_diff > 0:
             score += (total_diff_reduction / initial_total_diff) * 40
        score -= len(rule.operations) * 1
        return score, final_diff if final_diff is not None else initial_total_diff

    def generate(self, examples, num_hypotheses=100):
        """
        Main generation method.
        Args:
            examples (list): List of {'input': grid, 'output': grid} dictionaries.
            num_hypotheses (int): Max number of hypotheses to return.

        Returns:
            list: List of Rule objects, sorted by score (best first).
        """
        if not examples:
            return []

        # Use the first example to seed initial hypotheses, but evaluate on all
        seed_input = examples[0]['input']
        seed_output = examples[0]['output']

        # --- Step 1: Generate Single-Step Hypotheses ("Baby Problems") ---
        current_hypotheses = []
        single_step_rules = self._generate_single_step_hypotheses(seed_input, seed_output)
        
        print(f"Generated {len(single_step_rules)} initial single-step rules.")

        for rule in single_step_rules:
            score, final_diff = self._evaluate_and_score(rule, examples)
            if score > -float('inf'): # Check if rule was applicable at all
                # Store hypothesis with score and potential for refinement
                 # The 'remaining_diff_func' could be complex, maybe just store score for now
                 current_hypotheses.append({'rule': rule, 'score': score, 'depth': 1, 'final_diff': final_diff}) 
        
        # Sort initial hypotheses
        current_hypotheses.sort(key=lambda x: x['score'], reverse=True)
        
        completed_hypotheses = [h for h in current_hypotheses if h['final_diff'] == 0]
        partial_hypotheses = [h for h in current_hypotheses if h['final_diff'] > 0]


        # --- Step 2: Combine Partials (Iterative Deepening / Beam Search idea) ---
        # Keep track of the best hypotheses found so far
        best_hypotheses = current_hypotheses 

        for depth in range(1, self.max_depth):
            newly_generated = []
            # Take the most promising partial hypotheses from the previous depth
            # Use beam search: only expand the top K hypotheses
            beam_width = 5 
            candidates_to_expand = sorted([h for h in best_hypotheses if h['depth'] == depth and h['final_diff'] > 0], 
                                          key=lambda x: x['score'], reverse=True)[:beam_width]

            if not candidates_to_expand:
                break # No promising partials left to expand

            print(f"\nExpanding {len(candidates_to_expand)} hypotheses at depth {depth+1}")

            for hypo in candidates_to_expand:
                base_rule = hypo['rule']
                
                # Apply the base rule to get intermediate grids for *all* examples
                intermediate_grids = []
                try:
                    for ex in examples:
                        intermediate_grids.append(base_rule.apply(ex['input']))
                except Exception:
                     continue # Skip if base rule failed

                # Now, try adding ONE more operation
                # Generate single-step hypotheses based on the diff between *intermediate* and *final* output
                
                # Use the first example's intermediate state to generate next steps
                next_step_rules = self._generate_single_step_hypotheses(intermediate_grids[0], examples[0]['output'])

                for next_rule_step in next_step_rules:
                    if not next_rule_step.operations: continue # Skip empty rules
                    
                    # Combine: Append the new operation to the base rule's operations
                    combined_ops = base_rule.operations + next_rule_step.operations
                    
                    # Avoid overly long rules early on (optional)
                    # if len(combined_ops) > self.max_depth: continue

                    combined_rule = Rule(f"{base_rule.description} THEN {next_rule_step.description}", combined_ops)
                    
                    # Evaluate the *combined* rule
                    score, final_diff = self._evaluate_and_score(combined_rule, examples)

                    newly_generated.append({'rule': combined_rule, 'score': score, 'depth': depth + 1, 'final_diff': final_diff})

            # Add newly generated hypotheses and resort
            best_hypotheses.extend(newly_generated)
            best_hypotheses.sort(key=lambda x: x['score'], reverse=True)
            # Keep only top N overall hypotheses to manage complexity
            best_hypotheses = best_hypotheses[:max(num_hypotheses * 2, 20)] 

            # Update completed/partial lists (optional, mainly for inspection)
            completed_hypotheses = [h for h in best_hypotheses if h['final_diff'] == 0]
            partial_hypotheses = [h for h in best_hypotheses if h['final_diff'] > 0]
            print(f"Depth {depth+1}: Found {len(completed_hypotheses)} complete, {len(partial_hypotheses)} partial. Best score: {best_hypotheses[0]['score'] if best_hypotheses else 'N/A'}")


        # --- Step 3: Return Top Hypotheses ---
        # Select the best N hypotheses overall
        final_selection = sorted([h for h in best_hypotheses if h['score'] > -float('inf')], # Filter out failed rules
                                 key=lambda x: x['score'], reverse=True)

        print(f"\nGenerated {len(final_selection)} final hypotheses.")
        return [h['rule'] for h in final_selection[:num_hypotheses]]