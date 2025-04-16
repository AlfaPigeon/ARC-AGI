import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import threading

# For color mapping (simple palette for up to 10 unique values)
COLORS = [
    '#FFFFFF', '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
    '#00FFFF', '#FF00FF', '#C0C0C0', '#FFA500', '#A52A2A', '#000000'
]

# --- Matrix display constants ---
MATRIX_CANVAS_SIZE = 120  # px, reduced size for each matrix display
MATRIX_CANVAS_MIN_CELL = 16  # px, reduced minimum cell size
MATRIX_CANVAS_MAX_CELL = 24  # px, reduced maximum cell size

def get_color(val):
    try:
        return COLORS[int(val) % len(COLORS)]
    except:
        return '#CCCCCC'

class MatrixCanvas(tk.Canvas):
    def __init__(self, parent, matrix, **kwargs):
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        # Dynamically calculate cell size to fit the entire matrix within the max canvas size
        if rows > 0 and cols > 0:
            cell_size = min(
                MATRIX_CANVAS_MAX_CELL,
                max(MATRIX_CANVAS_MIN_CELL, min(MATRIX_CANVAS_SIZE // rows, MATRIX_CANVAS_SIZE // cols))
            )
        else:
            cell_size = MATRIX_CANVAS_MIN_CELL
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
                # Adjust font size to fit cell
                font_size = max(10, min(cell_size // 2, 18))
                self.create_text(
                    j*cell_size+cell_size//2, i*cell_size+cell_size//2,
                    text=str(val), font=('Arial', font_size)
                )

class ARCApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('ARC Solver & Output Visualizer')
        self.geometry('1200x750')
        self.configure(bg='#f4f4f4')

        # Solver controls
        self.train_dir = tk.StringVar()
        self.eval_dir = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.path.join('data', 'output'))
        self.use_preschool = tk.BooleanVar(value=False)
        self.data_subdirs = [os.path.join('data', d) for d in ['arc-prize-2025', 'evaluation', 'output', 'preschool', 'training']]

        # --- Layout ---
        top_frame = tk.Frame(self, bg='#f4f4f4')
        top_frame.pack(fill='x', padx=20, pady=10)
        form_font = ('Segoe UI', 11)
        label_opts = {'font': form_font, 'bg': '#f4f4f4'}
        entry_opts = {'font': form_font, 'bg': '#fff'}

        # Training dir
        tk.Label(top_frame, text='Training Directory:', **label_opts).grid(row=0, column=0, sticky='e', pady=4)
        self.train_dropdown = ttk.Combobox(top_frame, textvariable=self.train_dir, values=self.data_subdirs, width=38, font=form_font)
        self.train_dropdown.grid(row=0, column=1, sticky='w', padx=(0,8))
        tk.Button(top_frame, text='Browse', command=self.browse_train, font=form_font).grid(row=0, column=2, padx=2)

        # Eval dir
        tk.Label(top_frame, text='Evaluation Directory:', **label_opts).grid(row=1, column=0, sticky='e', pady=4)
        self.eval_dropdown = ttk.Combobox(top_frame, textvariable=self.eval_dir, values=self.data_subdirs, width=38, font=form_font)
        self.eval_dropdown.grid(row=1, column=1, sticky='w', padx=(0,8))
        tk.Button(top_frame, text='Browse', command=self.browse_eval, font=form_font).grid(row=1, column=2, padx=2)

        # Output dir
        tk.Label(top_frame, text='Output Directory:', **label_opts).grid(row=2, column=0, sticky='e', pady=4)
        tk.Entry(top_frame, textvariable=self.output_dir, width=42, **entry_opts).grid(row=2, column=1, sticky='w', padx=(0,8))
        tk.Button(top_frame, text='Browse', command=self.browse_output, font=form_font).grid(row=2, column=2, padx=2)

        # Preschool checkbox
        tk.Checkbutton(top_frame, text='Use Preschool Data', variable=self.use_preschool, command=self.set_preschool_dirs, font=form_font, bg='#f4f4f4').grid(row=3, column=1, sticky='w', pady=4)

        # Run button
        tk.Button(top_frame, text='Run Solver', command=self.run_solver, font=('Segoe UI', 12, 'bold'), bg='#0078d7', fg='white', relief='raised').grid(row=4, column=1, pady=10, sticky='w')

        # Progress bar and status
        self.progress = ttk.Progressbar(self, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(pady=(0, 8))
        self.status_var = tk.StringVar(value='Ready.')
        self.status_label = tk.Label(self, textvariable=self.status_var, font=('Segoe UI', 10), bg='#f4f4f4', fg='#333')
        self.status_label.pack(anchor='w', padx=22)
        # Precision label (added)
        self.precision_var = tk.StringVar(value='Precision: N/A')
        self.precision_label = tk.Label(self, textvariable=self.precision_var, font=('Segoe UI', 11, 'bold'), bg='#f4f4f4', fg='#0078d7')
        self.precision_label.pack(anchor='w', padx=22, pady=(0, 4))

        # --- Log and Matrix display ---
        main_frame = tk.Frame(self, bg='#f4f4f4')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        # Output file list for visualizer (move to leftmost)
        self.filebox = tk.Listbox(main_frame, width=40)
        self.filebox.grid(row=0, column=0, rowspan=4, sticky='ns', padx=(0, 10), pady=10)
        self.filebox.bind('<<ListboxSelect>>', self.on_file_select)
        self.refresh_file_list()

        # Output window (log) in the middle
        self.log = scrolledtext.ScrolledText(main_frame, width=60, height=28, state='disabled', font=('Consolas', 10), bg='#f9f9f9')
        self.log.grid(row=0, column=1, sticky='nsw', padx=(0, 16))

        # Matrix display area (right)
        self.matrix_frame = tk.Frame(main_frame, bg='#f4f4f4')
        self.matrix_frame.grid(row=0, column=2, sticky='n')
        self.input_label = tk.Label(self.matrix_frame, text='Input', font=('Segoe UI', 11, 'bold'), bg='#f4f4f4')
        self.input_label.grid(row=0, column=0, padx=10, pady=(0, 2))
        self.output_label = tk.Label(self.matrix_frame, text='Predicted Output', font=('Segoe UI', 11, 'bold'), bg='#f4f4f4')
        self.output_label.grid(row=0, column=1, padx=10, pady=(0, 2))
        self.gt_label = tk.Label(self.matrix_frame, text='Ground Truth', font=('Segoe UI', 11, 'bold'), bg='#f4f4f4')
        self.gt_label.grid(row=0, column=2, padx=10, pady=(0, 2))
        self.input_canvas = None
        self.output_canvas = None
        self.gt_canvas = None

    def refresh_file_list(self):
        output_folder = self.output_dir.get()
        self.filebox.delete(0, tk.END)
        if os.path.exists(output_folder):
            files = [f for f in os.listdir(output_folder) if f.endswith('.json')]
            for fname in files:
                self.filebox.insert(tk.END, fname)
        self.file_list = files if os.path.exists(output_folder) else []

    def on_file_select(self, event):
        selection = self.filebox.curselection()
        if not selection:
            return
        fname = self.file_list[selection[0]]
        fpath = os.path.join(self.output_dir.get(), fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            self.show_matrices(data.get('input', []), data.get('output', []), data.get('ground_truth', []))
            self.log_write(f'[VISUALIZER] Showing {fname}\n')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load {fname}: {e}')

    def set_preschool_dirs(self):
        if self.use_preschool.get():
            self.train_dir.set(os.path.join('data', 'preschool', 'training'))
            self.eval_dir.set(os.path.join('data', 'preschool', 'evaluation'))
        else:
            self.train_dir.set(os.path.join('data', 'training'))
            self.eval_dir.set(os.path.join('data', 'evaluation'))

    def browse_train(self):
        path = filedialog.askdirectory()
        if path:
            self.train_dir.set(path)

    def browse_eval(self):
        path = filedialog.askdirectory()
        if path:
            self.eval_dir.set(path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)
            self.refresh_file_list()

    def run_solver(self):
        train = self.train_dir.get()
        eval_ = self.eval_dir.get()
        output = self.output_dir.get()
        if not (train and eval_ and output):
            messagebox.showerror('Error', 'Please select all directories.')
            return
        threading.Thread(target=self._run_solver_thread, args=(train, eval_, output), daemon=True).start()

    def _run_solver_thread(self, train, eval_, output):
        self.log_clear()
        self.status_var.set('Loading training tasks...')
        self.progress['value'] = 0
        from solver import load_arc_tasks
        import general_inteligence
        train_tasks = load_arc_tasks(train)
        self.status_var.set('Loading evaluation tasks...')
        eval_tasks = load_arc_tasks(eval_)
        # Clean output directory
        if os.path.exists(output):
            for filename in os.listdir(output):
                file_path = os.path.join(output, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(output)
        self.log_write(f'[INFO] Loaded {len(train_tasks)} training tasks.\n')
        self.log_write(f'[INFO] Loaded {len(eval_tasks)} evaluation tasks.\n')
        self.status_var.set('Training solver...')
        solver = general_inteligence.ARCSolver()
        all_train_pairs = []
        for task in train_tasks.values():
            all_train_pairs.extend(task['train'])
        solver.fit({'train': all_train_pairs})
        self.log_write(f'[INFO] Best rule found: {str(getattr(solver, "best_rule", ""))}\n')
        total = 0
        correct = 0
        eval_task_list = list(eval_tasks.items())
        for idx, (task_id, task) in enumerate(eval_task_list):
            self.status_var.set(f'Processing Task {task_id} ({idx+1}/{len(eval_task_list)})...')
            self.progress['value'] = (idx+1) / len(eval_task_list) * 100
            self.log_write(f'\n[PROCESSING] Task {task_id}\n')
            predictions = solver.solve(task['test'])
            for i, (pair, pred) in enumerate(zip(task['test'], predictions)):
                self.log_write(f'\n[EVAL] Task {task_id} Test case {i}\n')
                gt = pair.get('output', None)
                self.show_matrices(pair['input'], pred, gt)
                self.log_write(f'Input: (see visual)\n')
                self.log_write(f'Predicted Output: (see visual)\n')
                if gt is not None:
                    self.log_write(f'Ground Truth: (see visual)\n')
                output_dict = {
                    "input": pair['input'].tolist() if hasattr(pair['input'], 'tolist') else pair['input'],
                    "output": pred.tolist() if hasattr(pred, 'tolist') else pred
                }
                if gt is not None:
                    output_dict["ground_truth"] = gt.tolist() if hasattr(gt, 'tolist') else gt
                os.makedirs(output, exist_ok=True)
                output_path = os.path.join(output, f"{task_id}_test{i}.json")
                with open(output_path, "w") as f:
                    json.dump(output_dict, f)
                if gt is not None:
                    match = np.array_equal(pred, gt)
                    self.log_write(f'Match: {match}\n')
                    correct += int(match)
                    total += 1
        precision = correct / total if total else 0
        self.status_var.set('Done.')
        self.progress['value'] = 100
        self.precision_var.set(f'Precision: {precision:.3f} ({correct}/{total})')
        self.log_write(f"\n[RESULT] Final Precision: {precision:.3f} ({correct}/{total})\n")
        self.refresh_file_list()

    def show_matrices(self, input_matrix, output_matrix, gt_matrix=None):
        # Remove old canvases if they exist
        for canvas in [self.input_canvas, self.output_canvas, self.gt_canvas]:
            if canvas:
                canvas.destroy()
        # Convert numpy arrays to lists if needed
        if hasattr(input_matrix, 'tolist'):
            input_matrix = input_matrix.tolist()
        if hasattr(output_matrix, 'tolist'):
            output_matrix = output_matrix.tolist()
        if gt_matrix is not None and hasattr(gt_matrix, 'tolist'):
            gt_matrix = gt_matrix.tolist()
        self.input_canvas = MatrixCanvas(self.matrix_frame, input_matrix)
        self.input_canvas.grid(row=1, column=0, padx=10, pady=8)
        self.output_canvas = MatrixCanvas(self.matrix_frame, output_matrix)
        self.output_canvas.grid(row=1, column=1, padx=10, pady=8)
        if gt_matrix is not None:
            self.gt_canvas = MatrixCanvas(self.matrix_frame, gt_matrix)
            self.gt_canvas.grid(row=1, column=2, padx=10, pady=8)
        else:
            self.gt_canvas = None

    def log_write(self, text):
        self.log.configure(state='normal')
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state='disabled')

    def log_clear(self):
        self.log.configure(state='normal')
        self.log.delete(1.0, tk.END)
        self.log.configure(state='disabled')

if __name__ == '__main__':
    app = ARCApp()
    app.mainloop()
