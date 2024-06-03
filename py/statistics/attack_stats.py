import os



def get_data(file_path, iteration):
    file_path = f"{file_path}/attack_success_object.txt"
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

iteration_list = [10,20,30,40]
model_list = ["DGCNN","GIN0","GIN0WithJK"]
# 初始化字典存储模型和对应的文件夹数量
model_counts = {iteration:{} for iteration in iteration_list }

for iteration in iteration_list:
    for model in model_list:
        print(f"Iteration: {iteration}  Model: {model}")
        base_folder=f"/home/lebron/IRFuzz/done_result/{iteration}/{model}/IRFuzz"
        get_data(file_path=base_folder, iteration=iteration)

print(model_counts)
exit()