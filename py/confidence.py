import os
import json
import argparse

from new_fuzz_attack import *


class Fuzz:
    
    def __init__(self, source_dir, fuzz_dir, model, mutator_counts, output_file_path, confidence_map):
        
        self.source_dir = source_dir
        self.fuzz_dir = fuzz_dir
        self.bash_sh = f"{source_dir}/fuzz_compile.sh"
        self.temp_bb_file_path = f"{fuzz_dir}/temp/.basicblock"
        self.model = model
        self.LDFLAGS, self.CFLAGS= read_bash_variables(f"{source_dir}/compile.sh")
        self.compiler = find_compilers(f"{source_dir}/compile.sh")
        self.function_probabilities = {}  # 添加字典以记录函数的概率
        self.iteration = 0
        self.mutator_counts = mutator_counts
        self.output_file_path = output_file_path
        self.confidence_map = confidence_map

        if self.source_dir  not in self.confidence_map[model]:
            self.confidence_map[self.model][self.source_dir] = {}
        
        self.fuzz_log = FuzzLog(fuzz_dir)
        self.fuzz_log.write(f"Model:{self.model}\n", "blue")
        
        build_fuzz_directories(self.fuzz_dir)
        
        self.bb_file_path =  f"{source_dir}/BasicBlock.txt"
        self.functions = parse_file(self.bb_file_path)
        
        # 示例输出,获取初始概率
        self.fuzz_log.write(f"初始概率为:", "green")
        self.temp_functions = self.functions
        # 将temp输出到temp目录中
        output_file(self.temp_functions, self.temp_bb_file_path)
        self.init_probability_0, self.init_probability_1 = self.get_probability()
        self.adversarial_label = 0 if self.init_probability_0 < self.init_probability_1 else 1 # 哪个概率小，哪个就是对抗样本标签
        self.fuzz_log.write(f"对抗样本label标签为:{self.adversarial_label}\n", "green")
        # print(self.init_probability_0, self.init_probability_1)
    
    def run(self):

        mutators = [
            ("random_block15+bcf30", None),
            ("random_block15+flatten3", None),
            ("bcf30+flatten3", None),
            ("random_block15+bcf30+flatten3", None),
            ("random_block", 1),
            ("random_block", 2),
            ("random_block", 3),
            ("random_block", 4),
            ("random_block", 5),
            ("random_block", 10),
            ("random_block", 15),
            ("random_block", 30),
            ("random_block", 40),
            ("random_block", 50),
            ("bcf",1),
            ("bcf",2),
            ("bcf",3),
            ("bcf",4),
            ("bcf",5),
            ("flatten",1),
            ("flatten",2),
            ("flatten",3),
            ("flatten",4),
            ("flatten",5),
        ]
        
        
        # 最初对抗标签的概率
        previous_adv_probability = self.init_probability_0 if self.adversarial_label == 0 else self.init_probability_1
        self.confidence_map[self.model][self.source_dir]["init"] = previous_adv_probability
        
        copy_file_to_folder(source_file=f"{self.source_dir}/BasicBlock.txt",target_folder=f"{self.fuzz_dir}/in")
        self.file_hashes = parse_hash_file(f"{self.fuzz_dir}/in")
        self.seed_list = [SeedFile(f) for f in list_seed_files(directory=f"{self.fuzz_dir}/in")]
        self.seed_count = len(self.seed_list) - 1
        self.fuzz_log.write(f"there is {self.seed_count} seed files\n","green")

        for (chosen_mutator, num) in mutators:
            # 如果跑过了就跳过
            if f"{chosen_mutator}_{num}" in self.confidence_map[self.model][self.source_dir]:
                continue
            # 重新解析初始种子
            seed_file = self.seed_list[0]
            self.fuzz_log.write(f"Selected seed file: {seed_file.path} with energy {seed_file.energy}\n", "blue")
            functions = parse_file(seed_file.path)              # 解析原函数文件
            
            self.fuzz_log.write(f"Chosen mutator: {chosen_mutator}_{num}\n", "yellow")
            
            # 检查是否为组合 mutator
            if "+" in chosen_mutator:
                # 按 "+" 分隔 mutator 并依次调用
                components = chosen_mutator.split("+")
                for component in components:
                    # 提取 mutator 名称和参数
                    import re
                    match = re.match(r"([a-zA-Z_]+)(\d+)", component)
                    if match:
                        name, num = match.groups()
                        num = int(num)
                        if name == "random_block":
                            self.mutate_random_block(functions, num)
                        elif name == "flatten":
                            self.mutate_flatten(functions, num)
                        elif name == "bcf":
                            self.mutate_bcf(functions, num)
            else:
                # 单独调用的逻辑（原有逻辑）
                if chosen_mutator == "random_block":
                    self.mutate_random_block(functions, num)
                elif chosen_mutator == "flatten":
                    self.mutate_flatten(functions, num)
                elif chosen_mutator == "bcf":
                    self.mutate_bcf(functions, num)
            
            self.temp_functions = functions                    
            output_file(self.temp_functions, self.temp_bb_file_path)    # 将temp输出到temp目录中
            probability_0, probability_1 = self.get_probability()       # 模型预测概率变化
            adversarial_probability = probability_0 if self.adversarial_label == 0 else probability_1 # 获取对抗样本标签
            
            self.confidence_map[self.model][self.source_dir][f"{chosen_mutator}_{num}"] = adversarial_probability

            # 实时保存结果
            with open(self.output_file_path, "w", encoding="utf-8") as f:
                json.dump(self.confidence_map, f, indent=4, ensure_ascii=False) 
        
        return 0
    
    def get_probability(self):
        # 插入+链接
        res = run_bash(script_path= self.bash_sh,
                args=[self.source_dir, self.fuzz_dir, self.temp_bb_file_path, self.LDFLAGS, self.CFLAGS, self.compiler])
        if res == -1:
            print("run fuzz_compile.sh failed! Please check carefully!\n")
            exit()
            
        # 返汇编
        disassemble(fuzz_dir=self.fuzz_dir)
        # 提取cfg
        extract_cfg(fuzz_dir=self.fuzz_dir)
        # 模型预测
        next_state, result, prediction, before_classifier_output = measure(self.fuzz_dir, model=self.model) # prediction 0是良性 1是恶意  目前要把恶意转为良性。 result是模型输出的logsoftmax概率
        result = torch.exp(result) # 将模型输出的logsoftmax转换为softmax
        formatted_tensor = torch.tensor([[float("{:f}".format(result[0][0])), float("{:f}".format(result[0][1]))]], requires_grad=True)
        probability_0 = formatted_tensor.tolist()[0][0] # 暂时先是一个样本
        probability_1 = formatted_tensor.tolist()[0][1] # 暂时先是一个样本
        self.fuzz_log.write(f"probability_0 is {probability_0} probability_1:{probability_1} \n\n", "green")

        return probability_0,  probability_1

    def mutate_random_block(self, functions, num):
        for i in range(num):
            functionName = random.choice(list(functions.keys()))
            # 选择一个随机块进行操作    
            blockNum = random.choice(list(functions[functionName].blocks.keys()))
            asmIndex = random.randint(0, 26)
            functions[functionName].blocks[blockNum].asm_indices.append(asmIndex)
            self.fuzz_log.write(f"Mutated {functionName} at block {blockNum} with new asmIndex {asmIndex} \n", "magenta")
        return functionName

    def mutate_flatten(self, functions, num):
        for i in range(num):
            # 随机选择函数并增加 flatten 次数
            functionName = random.choice(list(functions.keys()))
            if functions[functionName].flatten_level == 3:
                pass
            else:
                functions[functionName].flatten_level += 1
            self.fuzz_log.write(f"Increased flatten level for {functionName} to {functions[functionName].flatten_level}\n", "magenta")
        return functionName
    
    def mutate_bcf(self, functions, num):
        for i in range(num):
            functionName = random.choice(list(functions.keys()))
            # 随机选择函数并增加 bcf 概率
            functionName = random.choice(list(functions.keys()))
            functions[functionName].bcf_rate += 10
            self.fuzz_log.write(f"Increased bcf rate for {functionName} to {functions[functionName].bcf_rate}\n", "magenta")
        return functionName


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--model_list', type=str, nargs='+', default=["DGCNN_9", "DGCNN_20", "GIN0_9", "GIN0_20", "GIN0WithJK_9", "GIN0WithJK_20"],
                        help='List of models to use')
    args = parser.parse_args()
    model_list = args.model_list

    # 加载已存在的概率数据
    output_file_path = f"/home/lebron/confidence/{model_list[0]}_confidence_output.json"
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as f:
            confidence_map = json.load(f)
    else:
        confidence_map = {model: {} for model in model_list}

    # 配置相关参数
    malware_store_path = "/home/lebron/IRFuzz/ELF"
    malware_full_paths = [os.path.join(malware_store_path, entry) for entry in os.listdir(malware_store_path)]
    total_iterations = len(model_list) * len(malware_full_paths)

    for model in model_list:
        if model not in confidence_map:
            confidence_map[model] = {}
        average_label_confidence = 0
        progressed = 0
        mutator_counts = {}
        for malware_dir in malware_full_paths:
            source_dir = malware_dir
            if source_dir in confidence_map[model] and len(confidence_map[model][source_dir]) == 25:
                progressed += 1
                continue
            fuzz_dir = malware_dir
            fuzz = Fuzz(source_dir, fuzz_dir, model, mutator_counts, output_file_path, confidence_map)
            fuzz.run()
            print("Now is process {:.2f}%".format((progressed / total_iterations) * 100))
            progressed += 1
    print("Processing completed.")
