import os
import json
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox
import numpy as np

# For color mapping (simple palette for up to 10 unique values)
COLORS = [
    '#FFFFFF', '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
    '#00FFFF', '#FF00FF', '#C0C0C0', '#FFA500', '#A52A2A', '#000000'
]

def get_color(val):
    try:
        return COLORS[int(val) % len(COLORS)]
    except:
        return '#CCCCCC'

class MatrixCanvas(tk.Canvas):
    def __init__(self, parent, matrix, cell_size=20, **kwargs):
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        width = cols * cell_size
        height = rows * cell_size
        super().__init__(parent, width=width, height=height, **kwargs)
        self.draw_matrix(matrix, cell_size)

    def draw_matrix(self, matrix, cell_size):
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                color = get_color(val)
                self.create_rectangle(
                    j*cell_size, i*cell_size, (j+1)*cell_size, (i+1)*cell_size,
                    fill=color, outline='gray'
                )
                self.create_text(
                    j*cell_size+cell_size//2, i*cell_size+cell_size//2,
                    text=str(val), font=('Arial', 8)
                )

class OutputVisualizer(tk.Tk):
    def __init__(self, output_folder):
        super().__init__()
        self.title('Output Visualizer')
        self.geometry('900x600')
        self.output_folder = output_folder
        self.file_list = self.get_json_files()
        self.create_widgets()

    def get_json_files(self):
        return [f for f in os.listdir(self.output_folder) if f.endswith('.json')]

    def create_widgets(self):
        self.filebox = tk.Listbox(self, width=40)
        for fname in self.file_list:
            self.filebox.insert(tk.END, fname)
        self.filebox.grid(row=0, column=0, rowspan=4, sticky='ns', padx=(10, 20), pady=10)
        self.filebox.bind('<<ListboxSelect>>', self.on_file_select)

        # Add a frame to hold the matrices and their labels
        self.matrix_frame = tk.Frame(self)
        self.matrix_frame.grid(row=0, column=1, sticky='n', padx=10, pady=10)

        self.input_label = ttk.Label(self.matrix_frame, text='Input', font=('Arial', 12, 'bold'))
        self.input_label.grid(row=0, column=0, padx=20, pady=(0, 5))
        self.output_label = ttk.Label(self.matrix_frame, text='Output', font=('Arial', 12, 'bold'))
        self.output_label.grid(row=0, column=1, padx=20, pady=(0, 5))
        self.gt_label = ttk.Label(self.matrix_frame, text='Ground Truth', font=('Arial', 12, 'bold'))
        self.gt_label.grid(row=0, column=2, padx=20, pady=(0, 5))

        self.input_canvas = None
        self.output_canvas = None
        self.gt_canvas = None

    def on_file_select(self, event):
        selection = self.filebox.curselection()
        if not selection:
            return
        fname = self.file_list[selection[0]]
        fpath = os.path.join(self.output_folder, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            self.show_matrices(data)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load {fname}: {e}')

    def show_matrices(self, data):
        # Remove old canvases if they exist
        for canvas in [self.input_canvas, self.output_canvas, self.gt_canvas]:
            if canvas:
                canvas.destroy()
        input_matrix = data.get('input', [])
        output_matrix = data.get('output', [])
        gt_matrix = data.get('ground_truth', [])
        # Add padding between matrices
        self.input_canvas = MatrixCanvas(self.matrix_frame, input_matrix)
        self.input_canvas.grid(row=1, column=0, padx=20, pady=10)
        self.output_canvas = MatrixCanvas(self.matrix_frame, output_matrix)
        self.output_canvas.grid(row=1, column=1, padx=20, pady=10)
        self.gt_canvas = MatrixCanvas(self.matrix_frame, gt_matrix)
        self.gt_canvas.grid(row=1, column=2, padx=20, pady=10)

if __name__ == '__main__':
    output_folder = os.path.join('data', 'output')
    if not os.path.exists(output_folder):
        print(f'Output folder not found: {output_folder}')
    else:
        app = OutputVisualizer(output_folder)
        app.mainloop()
