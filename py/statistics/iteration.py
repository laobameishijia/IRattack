import os
import re
import numpy as np

def extract_iterations_from_success_files(base_folder):
    # Dictionary to store iteration counts for each model
    model_iterations = {}

    # Traverse through all numbered folders
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder,folder)
        if os.path.isdir(folder_path) and folder.isdigit():
            out_folder = os.path.join(folder_path, 'out')
            if os.path.exists(out_folder) and os.path.isdir(out_folder):
                for file in os.listdir(out_folder):
                    # print(f"Checking file: {file}")  # Debug: Print each file name
                    match = re.match(r"success_((DGCNN|GIN0|GIN0WithJK)_\d+)_(\d+)\.txt", file)
                    if match:
                        model_name = match.group(1)
                        iterations = int(re.search(r"_(\d+)\.txt", file).group(1))
                        if model_name not in model_iterations:
                            model_iterations[model_name] = []
                        model_iterations[model_name].append(iterations)

    return model_iterations


def calculate_five_number_summary(model_iterations):
    summaries = {}
    for model, iterations in model_iterations.items():
        if iterations:
            min_val = np.min(iterations) + 1
            q1 = np.percentile(iterations, 25) + 1
            median = np.median(iterations) + 1 
            q3 = np.percentile(iterations, 75) + 1
            max_val = np.max(iterations) + 1
            summaries[model] = (min_val, q1, median, q3, max_val)
    return summaries

def print_summaries(summaries):
    for model, summary in summaries.items():
        print(f"{model}:\n  Min: {summary[0]} iterations\n  Q1: {summary[1]} iterations\n  Median: {summary[2]} iterations\n  Q3: {summary[3]} iterations\n  Max: {summary[4]} iterations\n")

# Example usage
iteration_list = [10,20,30,40]
model_list = ["DGCNN","GIN0","GIN0WithJK"]
for iteration in iteration_list:
    print(f"Iteration: {iteration}")
    for model in model_list:
        base_folder = f"/home/lebron/IRFuzz/done_result/{iteration}/{model}/IRFuzz/ELF"
        model_iterations = extract_iterations_from_success_files(base_folder)
        summaries = calculate_five_number_summary(model_iterations)
        print_summaries(summaries)


exit()  
# Test
base_folder = '/home/lebron/IRFuzz/done_result/10/DGCNN/IRFuzz/ELF'  # Replace with your base folder path
model_iterations = extract_iterations_from_success_files(base_folder)
summaries = calculate_five_number_summary(model_iterations)
print_summaries(summaries)
