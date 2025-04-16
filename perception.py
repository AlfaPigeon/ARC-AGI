import numpy as np
from scipy.ndimage import label, find_objects, center_of_mass, rotate
from collections import Counter
import itertools
import copy # For deep copying grids during transformations
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