import argparse
import os
import subprocess
import time

ida64_path = r"wine C:\\IDA\\ida64.exe"
script_path = r"/root/IRattack/py/model/src/dataset_construct/disassemble.py"


import tqdm

def check_file_extension(filename, extensions):

    return any(filename.endswith(ext) for ext in extensions)

def check_file_in_folder(file_name, folder_path):
    
    # 获取文件夹中的所有文件名
    files_in_folder = os.listdir(folder_path)
    
    # 检查目标文件名是否在文件夹中
    return file_name in files_in_folder

def run(dir_path, output_dir, log_path):
    for file_name in tqdm.tqdm(os.listdir(dir_path)):
        if check_file_extension(file_name, ['.i64','.id0',".id1","id2",".nam",".til"]):
            continue
        if check_file_in_folder(file_name+".asm", f"{output_dir}"):
            continue
        file_path = dir_path +"/"+ file_name
        print(file_path)
        cmd = '{0} -L{1} -c -A -S"{2} {3} {4}" {5}'\
            .format(ida64_path, log_path, script_path, output_dir, file_name, file_path)
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True,text=True, timeout=30)
                if result.returncode == 0:
                    return 0  # 成功运行
                else:
                    print(f"Script failed with return code {result.returncode}. Retrying...")                
            except subprocess.TimeoutExpired:
                print("Disassemble timed out! Retrying...")
            retry_count += 1
            time.sleep(5)  # 等待一段时间后重试
            
        # print("STDOUT", result.stdout)
        # print("STDERR", result.stderr)
        # p = subprocess.Popen(cmd)
        # p.wait()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='data dir')
    parser.add_argument('-s', '--store_dir', type=str, required=True, help='store cfg')
    parser.add_argument('-l', '--log_file', type=str, required=True, help='log file')
    
    args = parser.parse_args()
    dir_path = args.data_dir
    output_dir = args.store_dir
    log_path = args.log_file
    run(dir_path, output_dir, log_path)
    