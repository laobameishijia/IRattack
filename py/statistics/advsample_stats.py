

import glob
import os
import re
import subprocess
import time

from termcolor import colored


def is_success_file_present(fuzz_dir, model):
    success_files = glob.glob(f"{fuzz_dir}/out/success_{model}*")
    return len(success_files) > 0     

def read_bash_variables(file_path):
    # 正则表达式用于匹配变量赋值，考虑了可能的空格和使用双引号
    pattern = r'^\s*(LDFLAGS|CFLAGS)\s*=\s*"([^"]*)"'
    variables = {'LDFLAGS': None, 'CFLAGS': None}
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(pattern, line)
                if match:
                    variables[match.group(1)] = match.group(2)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except IOError:
        print(f"Error: An error occurred while reading the file {file_path}.")
    
    return (variables["LDFLAGS"],variables["CFLAGS"])

def find_compilers(file_path):
    
    # Open and read the script file
    with open(file_path, 'r') as file:
        script_content = file.read()

    # Check for the presence of g++ and gcc in the content
    if "g++" in script_content:
        return "clang++"

    return "clang"
    
def run_bash(script_path, args, max_retries=5, retry_delay=5):
    retry_count = 0
    while retry_count < max_retries:
        try:
            result = subprocess.run([script_path] + args, timeout=30, text=True)
            if result.returncode == 0:
                return 0  # 成功运行
            else:
                print(f"Script failed with return code {result.returncode}. Retrying...")
        except subprocess.TimeoutExpired:
            print("The script timed out! Retrying...")

        retry_count += 1
        time.sleep(retry_delay)  # 等待一段时间后重试
    
    print("Max retries reached. Script failed to run successfully.")
    return -1  # 超过最大重试次数，仍然失败

def get_model_names(folder_path: str):
    # 定义正则表达式模式以匹配文件名并提取模型名称
    pattern = re.compile(r'success_([^_]+_[0-9]+)_')
    # 初始化结果列表
    result = []
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否以 success 开头
        if filename.startswith('success'):
            # 使用正则表达式匹配文件名并提取模型名称
            match = pattern.search(filename)
            if match:
                model_name = match.group(1)
                full_path = os.path.join(folder_path, filename)
                result.append((full_path, model_name))
    
    return result

if __name__ == "__main__":
    
    ATTACK_SUCCESS_MAP = {
        "DGCNN_9":[],
        "DGCNN_20":[],
        "GIN0_9":[],
        "GIN0_20":[],
        "GIN0WithJK_9":[],
        "GIN0WithJK_20":[]
    }
    ATTACK_SUCCESS_RATE = dict()

    malware_store_path = "/home/lebron/IRFuzz/ELF"
    advsample_store_path = "/home/lebron/IRFuzz/advsample"
    static_bash="/home/lebron/IRFuzz/bash/adv_sample_stats.sh"
    
    malware_full_paths = [os.path.join(malware_store_path, entry) for entry in os.listdir(malware_store_path)]

    for malware_dir in malware_full_paths:
        output_dir= advsample_store_path
        fuzz_dir=  malware_dir
        LDFLAGS, CFLAGS= read_bash_variables(f"{fuzz_dir}/compile.sh")
        compiler = find_compilers(f"{fuzz_dir}/compile.sh")
        success_model_list = get_model_names(f"{fuzz_dir}/out")
        # if os.path.exists(f"{fuzz_dir}/out/success_{model}.txt"):

            
        for basicblockfilepath, model in success_model_list:
            if is_success_file_present(fuzz_dir,model):
                print(colored(f"Already Attack Success! Next One!", "green"))
                ATTACK_SUCCESS_MAP[model].append(fuzz_dir.split('/')[-1])
            if os.path.exists(f"{fuzz_dir}/out/failed_{model}.txt"):
                print(colored(f"Already Failed!", "yellow"))
            run_bash(script_path=static_bash,args=[output_dir,fuzz_dir, basicblockfilepath, LDFLAGS,CFLAGS,compiler, model])

    for key in ATTACK_SUCCESS_MAP:
        ATTACK_SUCCESS_RATE[key] = len(ATTACK_SUCCESS_MAP[key]) / len(malware_full_paths)
    
    with open('/home/lebron/IRFuzz/attack_success_object.txt', 'w') as file:
        for key, object in ATTACK_SUCCESS_MAP.items():
            file.write(f'{key}: {str(object)}\n')  # 输出格式化的浮点数   
    
    with open('/home/lebron/IRFuzz/attack_success_rate.txt', 'w') as file:
        for key, value in ATTACK_SUCCESS_RATE.items():
            file.write(f'{key}: {value:.4f}\n')  # 输出格式化的浮点数
    exit()
    # /home/lebron/MalwareSourceCode-2/真正用C写的/Pass_Mirai-Iot-BotNet/IRattack/loader