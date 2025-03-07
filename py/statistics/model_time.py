import re
import numpy as np

def extract_model_times(filename):
    with open(filename, 'r') as file:
        lines = file.read()

    # Extract model list
    model_list_pattern = re.compile(r"model_list:\[(.+?)\]")
    model_list_match = model_list_pattern.search(lines)
    model_list = [model.strip().strip("'") for model in model_list_match.group(1).split(", ")] if model_list_match else []

    # Initialize a dictionary to store times for each model
    model_times = {model: [] for model in model_list}

    # General pattern to match model run times
    time_pattern = re.compile(r"(\w+)-\d+\nUse (\d+\.\d+) s")
    matches = time_pattern.findall(lines)
    for match in matches:
        model, time = match
        if model in model_times:
            model_times[model].append(float(time))

    # Calculate five-number summary for each model
    summaries = {}
    for model, times in model_times.items():
        if times:
            min_time = np.min(times)
            q1 = np.percentile(times, 25)
            median = np.median(times)
            q3 = np.percentile(times, 75)
            max_time = np.max(times)
            summaries[model] = (min_time, q1, median, q3, max_time)

    return summaries

def print_summaries(summaries):
    for model, summary in summaries.items():
        print(f"{model}:\n  Min: {summary[0]} s\n  Q1: {summary[1]} s\n  Median: {summary[2]} s\n  Q3: {summary[3]} s\n  Max: {summary[4]} s\n")

# Example usage
iteration_list = [10,20,30,40,50,60]
model_list = ["DGCNN","GIN0","GIN0WithJK"]
for iteration in iteration_list:
    print(f"Iteration: {iteration}")
    for model in model_list:
        filename=fr"F:\研二下\论文\备份\退修\result_2\{iteration}\{model}\IRFuzz\log"
        summaries = extract_model_times(filename)
        print_summaries(summaries)
exit()
# iteration_list = [10,20,30,40,50,60]
# model_list = ["DGCNN","GIN0","GIN0WithJK"]
# rerun_list = [40,50,60]
# rerun_model = ["GIN0_20"]
# for iteration in iteration_list:
#     print(f"Iteration: {iteration}")
#     for model in model_list:
#         if model in rerun_model and iteration in rerun_list:
#             filename=fr"F:\研二下\论文\备份\\125_done_result_new\done_result_new\{iteration}\{model}\IRFuzz\log"
#         else:   
#             filename=fr"F:\研二下\论文\备份\\126_done_result_new\done_result_new\{iteration}\{model}\IRFuzz\log"
#         summaries = extract_model_times(filename)
#         print_summaries(summaries)
# exit()
# Test
filename = '/home/lebron/IRFuzz/done_result/10/DGCNN/IRFuzz/log'
summaries = extract_model_times(filename)
print_summaries(summaries)
