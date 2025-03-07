# 实验脚本

```python
docker load --input image.tar 
docker run -it --gpus all --name 容器名 镜像名
docker exec -it 容器名 /bin/bash

cd ./IRattack
python3 ./py/new_fuzz_attack.py --max_iterations 10 --model_list 模型名

```

SRLAttack 和 IMalerAttack的复现，在python_new_pass分支下面。
