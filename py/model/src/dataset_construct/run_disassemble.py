import argparse
import os
import subprocess

ida64_path = r"wine /home/lebron/.wine/drive_c/Program\ Files/IDA_Pro_7.7/ida64.exe"
script_path = r"/home/lebron/MCFG_GNN/src/dataset_construct/disassemble.py"


import tqdm

def run(dir_path, output_dir, log_path):
    for file_name in tqdm.tqdm(os.listdir(dir_path)):
        # if file_name != "where2.exe": continue
        if file_name.endswith(".i64"): continue # 跳过那些.i64属于ida产生的中间过程文件
        file_path = dir_path +"/"+ file_name
        print(file_path)
        cmd = '{0} -L{1} -c -A -S"{2} {3} {4}" {5}'\
            .format(ida64_path, log_path, script_path, output_dir, file_name, file_path)
        result = subprocess.run(cmd, shell=True, capture_output=True,text=True)
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
    run(dir_path=dir_path,output_dir=output_dir,log_path=log_path)
    
# dir_path = r"E:\dataset\begnin"
# output_dir = r"E:\dataset\begnin_disassemble"
# log_path = r"E:\dataset\begnin_disassemble.log"
# run()

# dir_path = r"E:\dataset\malware"
# output_dir = r"E:\dataset\malware_disassemble"
# log_path = r"E:\dataset\malware_disassemble.log"
# run()


