import os
import re
import numpy as np

def extract_iterations_from_success_files(base_folder):
    # Dictionary to store iteration counts for each model
    model_iterations = {}

    # Traverse through all numbered folders
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path) and folder.isdigit():
            out_folder = os.path.join(folder_path, 'out')
            if os.path.exists(out_folder) and os.path.isdir(out_folder):
                for file in os.listdir(out_folder):
                    match = re.match(r"success_(.+?)_(\d+)", file)
                    if match:
                        model_name = match.group(1)
                        iterations = int(match.group(2))
                        if model_name not in model_iterations:
                            model_iterations[model_name] = []
                        model_iterations[model_name].append(iterations)

    return model_iterations

def calculate_five_number_summary(model_iterations):
    summaries = {}
    for model, iterations in model_iterations.items():
        if iterations:
            min_val = np.min(iterations)
            q1 = np.percentile(iterations, 25)
            median = np.median(iterations)
            q3 = np.percentile(iterations, 75)
            max_val = np.max(iterations)
            summaries[model] = (min_val, q1, median, q3, max_val)
    return summaries

def print_summaries(summaries):
    for model, summary in summaries.items():
        print(f"{model}:\n  Min: {summary[0]} iterations\n  Q1: {summary[1]} iterations\n  Median: {summary[2]} iterations\n  Q3: {summary[3]} iterations\n  Max: {summary[4]} iterations\n")

# Example usage
base_folder = 'path/to/base/folder'  # Replace with your base folder path
model_iterations = extract_iterations_from_success_files(base_folder)
summaries = calculate_five_number_summary(model_iterations)
print_summaries(summaries)
