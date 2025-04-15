import os
import json
import numpy as np
from glob import glob
import GeneralInteligence

TRAIN_DIR = os.path.join("data", "training")
EVAL_DIR = os.path.join("data", "evaluation")

def load_arc_tasks(folder):
    tasks = {}
    for path in glob(os.path.join(folder, "*.json")):
        with open(path, "r") as f:
            data = json.load(f)
        # Convert lists to np.arrays for 'input' and 'output'
        for phase in ['train', 'test']:
            for pair in data.get(phase, []):
                pair['input'] = np.array(pair['input'])
                pair['output'] = np.array(pair['output'])
        tasks[os.path.splitext(os.path.basename(path))[0]] = data
    return tasks

class Solver(GeneralInteligence.ARCSolver):
    def __init__(self, name: str = "Solver"):
        super().__init__()
        self.name = name
        self.solver = GeneralInteligence.ARCSolver()

    def fit(self, train_tasks):
        print("Training on {} tasks...".format(len(train_tasks)))
        for task_id, task in train_tasks.items():
            print(f"\n[TRAIN] Task {task_id}")
            self.solver.fit(task['train'])

    def evaluate(self, eval_tasks, verbose=True):
        total = 0
        correct = 0
        for task_id, task in eval_tasks.items():
            print(f"\n[EVAL] Task {task_id}")
            for i, pair in enumerate(task['test']):
                inp = pair['input']
                gt = pair.get('output', None)
                pred = self.solver.solve(inp)
                if verbose:
                    print(f"Test case {i}:")
                    print("Input:\n", inp)
                    print("Predicted Output:\n", pred)
                    if gt is not None:
                        print("Ground Truth:\n", gt)
                if gt is not None:
                    match = np.array_equal(pred, gt)
                    print("Match:", match)
                    correct += int(match)
                    total += 1
        precision = correct / total if total else 0
        print(f"\nFinal Precision: {precision:.3f} ({correct}/{total})")
        return precision

if __name__ == "__main__":
    print("Loading training tasks...")
    train_tasks = load_arc_tasks(TRAIN_DIR)
    print("Loading evaluation tasks...")
    eval_tasks = load_arc_tasks(EVAL_DIR)

    # Train on all training tasks
    solver = GeneralInteligence.ARCSolver()
    all_train_pairs = []
    for task in train_tasks.values():
        all_train_pairs.extend(task['train'])
    solver.fit({'train': all_train_pairs})

    total = 0
    correct = 0
    verbose = True

    # Evaluate on all evaluation tasks
    for task_id, task in eval_tasks.items():
        print(f"\n[PROCESSING] Task {task_id}")
        predictions = solver.solve(task['test'])
        for i, (pair, pred) in enumerate(zip(task['test'], predictions)):
            print(f"\n[EVAL] Task {task_id} Test case {i}")
            gt = pair.get('output', None)
            if verbose:
                print("Input:\n", pair['input'])
                print("Predicted Output:\n", pred)
                if gt is not None:
                    print("Ground Truth:\n", gt)
            # Save input, predicted output, and ground truth to JSON
            output_dict = {
                "input": pair['input'].tolist() if hasattr(pair['input'], 'tolist') else pair['input'],
                "output": pred.tolist() if hasattr(pred, 'tolist') else pred
            }
            if gt is not None:
                output_dict["ground_truth"] = gt.tolist() if hasattr(gt, 'tolist') else gt
            output_dir = os.path.join("data", "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{task_id}_test{i}.json")
            with open(output_path, "w") as f:
                json.dump(output_dict, f)
            if gt is not None:
                match = np.array_equal(pred, gt)
                print("Match:", match)
                correct += int(match)
                total += 1
    precision = correct / total if total else 0
    print(f"\nFinal Precision: {precision:.3f} ({correct}/{total})")