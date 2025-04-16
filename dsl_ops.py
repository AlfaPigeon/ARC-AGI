import numpy as np

def op_move_all_objects(grid, dr, dc, **kwargs):
    """Moves all non-background pixels by (dr, dc)."""
    new_grid = grid.copy()
    new_grid[:] = 0
    rows, cols = np.where(grid != 0) # Find all non-background pixels
    colors = grid[rows, cols]
    new_rows, new_cols = rows + dr, cols + dc
    # Filter points that move out of bounds
    valid_indices = (new_rows >= 0) & (new_rows < grid.shape[0]) & \
                    (new_cols >= 0) & (new_cols < grid.shape[1])
    new_grid[new_rows[valid_indices], new_cols[valid_indices]] = colors[valid_indices]
    return new_grid

def op_move_object(grid, object_masks, obj_id, dr, dc):
    """Moves a specific object by (dr, dc). Returns modified grid."""
    new_grid = grid.copy()
    if obj_id not in object_masks:
        return new_grid # Object not found
    mask = object_masks[obj_id]
    original_color = grid[mask][0] # Assumes uniform color
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
    new_grid = grid.copy()
    if obj_id not in object_masks:
        return new_grid
    new_grid[object_masks[obj_id]] = new_color
    return new_grid

def op_fill_rect(grid, top, left, bottom, right, color):
    """Fills a rectangular area."""
    new_grid = grid.copy()
    new_grid[top:bottom+1, left:right+1] = color
    return new_grid

def op_rotate_object(grid, object_masks, obj_id, angle, center):
    """ Rotates an object around a center point. COMPLEX to implement perfectly with grid pixels"""
    new_grid = grid.copy()
    print(f"Warning: op_rotate_object is complex and not fully implemented here.")
    return new_grid

def op_flip_horizontal(grid):
    """Flips the grid horizontally (left-right)."""
    new_grid = grid.copy()
    new_grid = np.fliplr(new_grid)
    return new_grid

def op_flip_vertical(grid):
    """Flips the grid vertically (up-down)."""
    new_grid = grid.copy()
    new_grid = np.flipud(new_grid)
    return new_grid

def op_rotate_90(grid):
    """Rotates the grid 90 degrees clockwise."""
    new_grid = grid.copy()
    new_grid = np.rot90(new_grid, k=-1)
    return new_grid

def op_rotate_180(grid):
    """Rotates the grid 180 degrees."""
    new_grid = grid.copy()
    new_grid = np.rot90(new_grid, k=2)
    return new_grid

def op_erase_object(grid, object_masks, obj_id):
    """Erases a specific object from the grid."""
    new_grid = grid.copy()
    if obj_id not in object_masks:
        return new_grid
    new_grid[object_masks[obj_id]] = 0
    return new_grid

def op_draw_line(grid, x0, y0, x1, y1, color):
    """Draws a straight line from (x0, y0) to (x1, y1) with the given color (Bresenham's algorithm)."""
    new_grid = grid.copy()
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                new_grid[x, y] = color
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                new_grid[x, y] = color
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    if 0 <= x1 < grid.shape[0] and 0 <= y1 < grid.shape[1]:
        new_grid[x1, y1] = color
    return new_grid

def op_invert_colors(grid):
    """Inverts all colors in the grid (0 <-> 9, 1 <-> 8, etc)."""
    new_grid = grid.copy()
    new_grid = 9 - new_grid
    return new_grid

def op_threshold(grid, threshold, **kwargs):
    """Binarizes the grid: values > threshold become 1, else 0."""
    new_grid = grid.copy()
    new_grid = (new_grid > threshold).astype(int)
    return new_grid

def op_blur(grid):
    """Applies a simple mean blur filter to the grid."""
    new_grid = grid.copy()
    from scipy.ndimage import uniform_filter
    new_grid = uniform_filter(new_grid, size=3, mode='constant')
    return new_grid

def op_draw_rectangle(grid, top, left, bottom, right, color):
    """Draws a rectangle border with the given color."""
    new_grid = grid.copy()
    new_grid[top, left:right+1] = color
    new_grid[bottom, left:right+1] = color
    new_grid[top:bottom+1, left] = color
    new_grid[top:bottom+1, right] = color
    return new_grid

def op_draw_circle(grid, center_x, center_y, radius, color):
    """Draws a circle with the given center, radius, and color."""
    new_grid = grid.copy()
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if abs((x - center_x)**2 + (y - center_y)**2 - radius**2) <= radius:
                new_grid[x, y] = color
    return new_grid

def op_replace_color(grid, old_color, new_color):
    """Replaces all instances of old_color with new_color."""
    new_grid = grid.copy()
    new_grid[grid == old_color] = new_color
    return new_grid

def op_count_color(grid, color):
    """Counts the number of cells with the given color."""
    new_grid = grid.copy()
    count = int(np.sum(new_grid == color))
    return count

def op_crop(grid, top, left, bottom, right):
    """Crops the grid to the specified rectangle."""
    new_grid = grid.copy()
    new_grid = new_grid[top:bottom+1, left:right+1]
    return new_grid

# DSL operations dictionary

dsl_ops = {
    'op_move_object': op_move_object,
    'op_recolor_object': op_recolor_object,
    'op_fill_rect': op_fill_rect,
    'op_move_all_objects': op_move_all_objects,
    'op_flip_horizontal': op_flip_horizontal,
    'op_flip_vertical': op_flip_vertical,
    'op_rotate_90': op_rotate_90,
    'op_rotate_180': op_rotate_180,
    'op_erase_object': op_erase_object,
    'op_draw_line': op_draw_line,
    'op_invert_colors': op_invert_colors,
    'op_threshold': op_threshold,
    'op_blur': op_blur,
    'op_draw_rectangle': op_draw_rectangle,
    'op_draw_circle': op_draw_circle,
    'op_replace_color': op_replace_color,
    'op_count_color': op_count_color,
    'op_crop': op_crop,
}
