
from collections import OrderedDict
import json
import os
import re


def parse_asm_indices(indices_str):
    """解析汇编指令索引字符串，返回索引列表"""
    return [int(idx) for idx in indices_str.split('+') if idx.strip()]

def parse_file(file_path):
    """解析文件并填充包含多个函数信息的字典,现在包括Flatten层级和BCF比例"""
    functions = {}
    current_function = None

    with open(file_path, 'r') as file:
        for line in file:
            if not line.strip():#空行跳过
                continue
            line = line.strip()
            if line.startswith('['):  # 新函数开始
                header = line.strip('[]')
                function_name, settings = header.split('@')
                flatten_level, bcf_rate = map(int, settings.split(','))
                
                if flatten_level not in mutation["flatten_level"]:
                    mutation["flatten_level"][flatten_level] = 1
                else:
                    mutation["flatten_level"][flatten_level] += 1
                
                if bcf_rate not in mutation["bcf_rate"]:
                    mutation["bcf_rate"][bcf_rate] = 1
                else:
                    mutation["bcf_rate"][bcf_rate] += 1     
                    
            else:
                function_name, rest = line.split('#')
                block_num_str, asm_part = rest.split('&')
                block_num = int(block_num_str)
                count_str, _, asm_indices_str = asm_part.partition(':')
                count = int(count_str.strip(':'))  # 修改此处，确保从字符串提取整数前去除了冒号和空格

                asm_indices = []
                if asm_indices_str.strip():  # 确保不是空字符串
                    asm_indices = parse_asm_indices(asm_indices_str)
                
                for asm_index in asm_indices:
                    if asm_index not in mutation["asm_index"]:
                        mutation["asm_index"][asm_index] = 1
                    else:
                        mutation["asm_index"][asm_index] += 1     
                    
                    
    file.close()
    
    return functions

def get_attack_success(folder_path, model):
    # 定义正则表达式模式以匹配文件名并提取模型名称和最后的数字
    pattern = re.compile(r'success_([^_]+_[0-9]+)_([0-9]+)\.txt')
    # 初始化结果列表
    # result = []
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否以 success 开头
        if filename.startswith('success'):
            # 使用正则表达式匹配文件名并提取模型名称
            match = pattern.search(filename)
            if match:
                model_name = match.group(1)
                if model_name == model:
                    full_path = os.path.join(folder_path, filename)
                    return (full_path, model_name)
    
    return None

def sort_dict_by_value(data: dict, reverse: bool = True) -> dict:
    """
    按照值的大小对字典进行排序，并返回一个有序字典。

    参数:
    data (dict): 需要排序的字典。
    reverse (bool): 是否按照降序排序。默认是 True (降序)。

    返回:
    dict: 排序后的有序字典。
    """
    # 使用 sorted 函数对字典按值排序
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=reverse)
    
    # 将排序后的项目转换为有序字典
    sorted_dict = OrderedDict(sorted_items)
    
    return sorted_dict


if __name__ == '__main__':
    
    iteration_list = [10,20,30,40]
    model_list = ["DGCNN","GIN0","GIN0WithJK"]
    for model in model_list:
        
        mutation = {
            'asm_index':{
                
            },
            'flatten_level': {
                
            },
            'bcf_rate':{
                
            },
        }
        
        sub_model_list = [f"{model}_9", f"{model}_20"]
        
        for sub_model in sub_model_list:
            for iteration in iteration_list:
                malware_store_path = f"/home/lebron/IRFuzz/done_result/{iteration}/{model}/IRFuzz/ELF"
                malware_full_paths = [os.path.join(malware_store_path, entry) for entry in os.listdir(malware_store_path)]
                success_model_list = []
                
                for malware_dir in malware_full_paths:
                    res = get_attack_success(f"{malware_dir}/out", sub_model)
                    if res != None:
                        success_model_list.append(res)
                
                for _ in success_model_list:
                    successfilepath, modelname = _[0],_[1]
                    parse_file(successfilepath)
                
                # 对每个子字典进行排序
                mutation = {k: sort_dict_by_value(v) for k, v in mutation.items()}    
                
            # 将字典写入 JSON 文件
            with open(f'/home/lebron/IRFuzz/stats/{sub_model}.json', 'w') as file:
                json.dump(mutation, file)




exit()

# Test
mutation = {
    'asm_index':{
        
    },
    'flatten_level': {
        
    },
    'bcf_rate':{
        
    },
    'itertion':[]
}
malware_store_path = "/home/lebron/IRFuzz/ELF"
malware_full_paths = [os.path.join(malware_store_path, entry) for entry in os.listdir(malware_store_path)]
model = "GIN0WithJK_9" # GIN0WithJK_9 && GIN0_9
success_model_list = []
for malware_dir in malware_full_paths:
    res = get_attack_success(f"{malware_dir}/out", model)
    if res != None:
        success_model_list.append(res)

for _ in success_model_list:
    successfilepath, modelname, itertion = _[0],_[1],_[2]
    parse_file(successfilepath)
    mutation['itertion'].append(itertion)

# 对每个子字典进行排序
mutation = {k: sort_dict_by_value(v) for k, v in mutation.items()}    

# 将字典写入 JSON 文件
with open(f'/home/lebron/IRFuzz/stats/{model}.json', 'w') as file:
    json.dump(mutation, file)
