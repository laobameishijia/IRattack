import json
import os

def get_data(file_path, iteration):
    file_path = f"{file_path}\\attack_success_object.txt"
    # 读取文件
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():  # 检查行是否为空
                # 分离模型名称和文件夹列表
                model, folders = line.split(':')
                # 解析文件夹列表
                folders = eval(folders.strip())
                # 统计文件夹数量
                folder_count = len(folders)
                # 存储结果
                model_counts[iteration][model] = folder_count

iteration_list = [10,20,30,40,50,60]
model_list = ["DGCNN_9","DGCNN_20","GIN0_9","GIN0_20","GIN0WithJK_9","GIN0WithJK_20"]
rerun_list = [40,50,60]
rerun_model = ["GIN0_20"]
# 初始化字典存储模型和对应的文件夹数量
model_counts = {iteration:{} for iteration in iteration_list }

for iteration in iteration_list:
    for model in model_list:
        print(f"Iteration: {iteration}  Model: {model}")
        model_name = model.split("_")[0]
        if model in rerun_model and iteration in rerun_list:
            base_folder=fr"F:\研二下\论文\备份\\125_done_result_new\done_result_new\{iteration}\{model_name}\IRFuzz"
        else:
            base_folder=fr"F:\研二下\论文\备份\\126_done_result_new\done_result_new\{iteration}\{model_name}\IRFuzz"
        get_data(file_path=base_folder, iteration=iteration)

print(json.dumps(model_counts, indent=4))

exit()