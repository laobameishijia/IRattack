import os
from collections import defaultdict

def count_basic_blocks(file_path: str) -> int:
    """
    统计 BasicBlock.txt 文件中的基本块数量。

    参数:
    file_path (str): BasicBlock.txt 文件的路径。

    返回:
    int: 基本块的数量。
    """
    basic_block_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # 检查是否是基本块描述行
            if line.strip() and  line.endswith(':') and not line.startswith('['):
                basic_block_count += 1
    return basic_block_count

def categorize_folders(base_folder: str):
    """
    遍历主文件夹下的所有子文件夹，统计每个子文件夹中的基本块数量，并分类。

    参数:
    base_folder (str): 主文件夹的路径。
    """
    folder_basic_block_counts = {}

    # 遍历所有子文件夹
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'BasicBlock.txt')
            if os.path.exists(file_path):
                basic_block_count = count_basic_blocks(file_path)
                folder_basic_block_counts[folder_name] = basic_block_count

    # 将子文件夹按照基本块数量排序
    sorted_folders = sorted(folder_basic_block_counts.items(), key=lambda x: x[1])

    # 分类
    total_folders = len(sorted_folders)
    categories = defaultdict(list)
    for index, (folder_name, count) in enumerate(sorted_folders):
        percentile = (index / total_folders) * 100
        if percentile <= 25:
            categories['0-25%'].append((folder_name, count))
        elif percentile <= 50:
            categories['25-50%'].append((folder_name, count))
        elif percentile <= 75:
            categories['50-75%'].append((folder_name, count))
        else:
            categories['75-100%'].append((folder_name, count))

    return categories

def print_categories(categories: dict):
    """
    打印分类结果。

    参数:
    categories (dict): 分类结果。
    """
    for category, folders in categories.items():
        print(f"{category}:")
        for folder_name, count in folders:
            print(f"  Folder: {folder_name}, Basic Block Count: {count}")

def calculate_success_rate(base_folder: str, categories: dict, model):
    """
    统计每个分类范围内的不同模型的攻击成功率。

    参数:
    base_folder (str): 主文件夹的路径。
    categories (dict): 分类后的文件夹信息。

    返回:
    dict: 每个分类范围内的不同模型的攻击成功率。
    """
    success_rates = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(lambda: defaultdict(int))

    for category, folders in categories.items(): 
        
        total_counts_file =  len(folders)
            
        for folder_name, _ in folders:
            out_folder_path = os.path.join(base_folder, folder_name, 'out')
            if os.path.exists(out_folder_path):
                for filename in os.listdir(out_folder_path):
                    if filename.startswith(f'success_{model}') and filename.endswith('.txt'):
                        parts = filename.split('_')
                        if len(parts) >= 3:
                            model_name = parts[1] + '_' + parts[2]
                            success_rates[category][model_name] += 1

    # 计算成功率
    for category in success_rates:
        for model_name in success_rates[category]:
            success_rates[category][model_name] = (success_rates[category][model_name] / len(categories[category])) * 100

    return success_rates

def print_success_rates(success_rates: dict):
    """
    打印每个分类范围内的不同模型的攻击成功率。

    参数:
    success_rates (dict): 每个分类范围内的不同模型的攻击成功率。
    """
    model_order = ["DGCNN_9", "DGCNN_20", "GIN0_9", "GIN0_20", "GIN0WithJK_9", "GIN0WithJK_20"]
    model_order = ["DGCNN_20", "GIN0_9", "GIN0_20", "GIN0WithJK_9"]
    
    for category, models in success_rates.items():
        print(f"{category}:")
        for model_name in model_order:
            if model_name in models:
                rate = models[model_name]
                print(f"  Model: {model_name}, Success Rate: {rate:.2f}%")

            
iteration_list = [10,20,30,40,50,60]
model_list = ["DGCNN_20","GIN0_9","GIN0_20","GIN0WithJK_9"]
for iteration in iteration_list:
    for model in model_list:
        print(f"Iteration: {iteration}  Model: {model}")
        model_name = model.split("_")[0]
        base_folder=fr"F:\研二下\论文\备份\退修\result_2\{iteration}\{model_name}\IRFuzz\ELF"
        categories = categorize_folders(base_folder)
        success_rates = calculate_success_rate(base_folder, categories, model)
        print_success_rates(success_rates)
    print("\n")
exit()

iteration_list = [10,20,30,40,50,60]
model_list = ["DGCNN_9","DGCNN_20","GIN0_9","GIN0_20","GIN0WithJK_9","GIN0WithJK_20"]
rerun_list = [40,50,60]
rerun_model = ["GIN0_20"]
for iteration in iteration_list:
    for model in model_list:
        print(f"Iteration: {iteration}  Model: {model}")
        model_name = model.split("_")[0]
        if model in rerun_model and iteration in rerun_list:
            base_folder=fr"F:\研二下\论文\备份\\125_done_result_new\done_result_new\{iteration}\{model_name}\IRFuzz\ELF"
        else:
            base_folder=fr"F:\研二下\论文\备份\\126_done_result_new\done_result_new\{iteration}\{model_name}\IRFuzz\ELF"
        categories = categorize_folders(base_folder)
        success_rates = calculate_success_rate(base_folder, categories, model)
        print_success_rates(success_rates)
    print("\n")
exit()
# 示例用法
base_folder = '/home/lebron/IRFuzz/done_result/10/DGCNN/IRFuzz/ELF'
categories = categorize_folders(base_folder)
# print_categories(categories)
success_rates = calculate_success_rate(base_folder, categories)
print_success_rates(success_rates)