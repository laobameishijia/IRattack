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
filename = 'log.txt'
summaries = extract_model_times(filename)
print_summaries(summaries)
