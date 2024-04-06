# created by Wu Bolun
# 2020.9.29
# bowenwu@sjtu.edu.cn

import argparse
import datetime
import os

from model.src.dataset_construct.asm import *

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 '{directory}' 不存在，已创建。")
    else:
        print(f"目录 '{directory}' 已存在。")

# 日志函数
def log(message, log_file='process.log'):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def run(data_dir, store_dir, file_format, log_file):
    
    create_directory_if_not_exists(store_dir)
    
    log_file = open(log_file, "a")
    current_time = datetime.datetime.now()
    log_file.write(current_time.strftime("%Y-%m-%d %H:%M:%S")  + "\n")
    log_file.write(f"Now dir is {data_dir} \n")
    
    count = 0
    file_list = os.listdir(data_dir)
    file_list = list(filter(lambda x: '.asm' in x, file_list))
    for filepath in file_list:
        count += 1
        # if '.asm' not in filepath:
        #     continue
        message = '{}/{} File: {}. Info: '.format(count, len(file_list), filepath)
        print('{}/{} File: {}. Info: '.format(count, len(file_list), filepath), end='')

        # binary_id = filepath.split('.')[0] # 无法兼容gawk-3.1.7.asm
        binary_id = filepath[:-4]
        
        # Unuseless
        # if binary_id in empty_code_ids:
        #     print('Empty code.')
        #     message = message + "Empty code.\n"
        #     log_file.write(message)
        #     continue

        store_path = os.path.join(store_dir, '{}.{}'.format(binary_id, file_format))
        if os.path.exists(store_path):
            print('Already parsed.')
            message = message + "Already parsed.\n"
            log_file.write(message)
            continue

        parser = AsmParser(directory=data_dir, binary_id=binary_id)
        success = parser.parse()
        if success:
            parser.store_blocks(store_path, fformat=file_format)
            print('Success.')
            message = message + "Success.\n"
            log_file.write(message)
        else:
            print('Empty code or block after parse.')
            message = message + "Empty code or block after parse.\n"
            log_file.write(message)
    
    log_file.close()

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='data dir')
    parser.add_argument('-s', '--store_dir', type=str, required=True, help='store cfg')
    parser.add_argument('-f', '--format', type=str, required=True, help='cfg file format')
    parser.add_argument('-l', '--log_file', type=str, required=True, help='log file')

    args = parser.parse_args()
    
    data_dir = args.data_dir
    store_dir = args.store_dir
    file_format = args.format
    log_file = args.log_file
    
    create_directory_if_not_exists(store_dir)
    
    # with open(r'F:\研二上\论文\dataset\empty_code.err', 'r') as f:
    #     empty_code_ids = f.read().split('\n')

    log_file = open(log_file, "a")
    current_time = datetime.datetime.now()
    log_file.write(current_time.strftime("%Y-%m-%d %H:%M:%S")  + "\n")
    log_file.write(f"Now dir is {data_dir} \n")
        
    count = 0
    file_list = os.listdir(data_dir)
    file_list = list(filter(lambda x: '.asm' in x, file_list))
    for filepath in file_list:
        count += 1
        # if '.asm' not in filepath:
        #     continue
        message = '{}/{} File: {}. Info: '.format(count, len(file_list), filepath)
        print('{}/{} File: {}. Info: '.format(count, len(file_list), filepath), end='')

        # binary_id = filepath.split('.')[0] # 无法兼容gawk-3.1.7.asm
        binary_id = filepath[:-4]
        
        # Unuseless
        # if binary_id in empty_code_ids:
        #     print('Empty code.')
        #     message = message + "Empty code.\n"
        #     log_file.write(message)
        #     continue

        store_path = os.path.join(store_dir, '{}.{}'.format(binary_id, file_format))
        if os.path.exists(store_path):
            print('Already parsed.')
            message = message + "Already parsed.\n"
            log_file.write(message)
            continue

        parser = AsmParser(directory=data_dir, binary_id=binary_id)
        success = parser.parse()
        if success:
            parser.store_blocks(store_path, fformat=file_format)
            print('Success.')
            message = message + "Success.\n"
            log_file.write(message)
        else:
            print('Empty code or block after parse.')
            message = message + "Empty code or block after parse.\n"
            log_file.write(message)
    
    log_file.close()
