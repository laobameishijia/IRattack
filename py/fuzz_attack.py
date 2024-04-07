import random

from model.src.dataset_construct import run_disassemble
from model.src.dataset_construct import run_extract_cfg
from model.src.gnn import run_measure

def main_loop(seed_file):
    seed_data = load_seed_file(seed_file)
    initial_prob = model_predict(seed_data) # 假设我们关注的分类概率为第一个类别

    for _ in range(NUM_ITERATIONS):
        mutation = random.choice(mutations)
        mutated_data = mutation(seed_data)
        new_prob = model_predict(mutated_data)

        # 假设我们关注的是将样本错误分类为第一个类别的概率
        if new_prob[0] > initial_prob[0]:
            # 如果错误分类的概率增加，则保存变异后的种子文件
            save_mutated_data(mutated_data)
            # 更新初始概率为新的更高的概率，以便之后的迭代可以基于这个新的种子文件
            initial_prob = new_prob

