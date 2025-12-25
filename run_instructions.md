# SeHGNN 项目运行指南

本项目是 AAAI 2023 论文 "Simple and Efficient Heterogeneous Graph Neural Network (SeHGNN)" 的官方实现。

## 1. 环境配置

### 基础依赖安装
首先确保安装了匹配 CUDA 版本的 PyTorch, PyTorch Geometric 和 DGL。

```bash
# 安装 requirements.txt 中的依赖
pip install -r requirements.txt
```

### 安装 sparse_tools
项目依赖于 `sparse_tools` 库，需要手动编译安装：

```bash
cd sparse_tools
python setup.py develop
cd ..
```

## 2. 数据准备

### HGB 数据集 (DBLP, ACM, IMDB, Freebase)
1. 从 [HGB repository](https://github.com/THUDM/HGB) 下载 `DBLP.zip`, `ACM.zip`, `IMDB.zip`, `Freebase.zip`。
2. 将它们解压到 `./data/` 目录下。

### Ogbn-mag 数据集
* 运行 `ogbn/main.py` 时会自动下载。
* 如果需要使用 ComplEx 嵌入（推荐），请参考下文的 OGBN-MAG 进阶运行部分。

## 3. 运行指令汇总

### HGB 数据集运行 (进入 `hgb/` 目录)

```bash
cd hgb
```
/home/jpf/miniconda3/envs/csx/bin/python /home/jpf/spz/SeHGNN-master/hgb/main.py --epoch 200 --dataset DBLP --n-fp-layers 2 --n-task-layers 3 --num-hops 2 --num-label-hops 4 --label-feats --residual --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --amp --seeds 1 2 3 4 5

*   **DBLP**:
    ```bash
    python main.py --epoch 200 --dataset DBLP --n-fp-layers 2 --n-task-layers 3 --num-hops 2 --num-label-hops 4 --label-feats --residual --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --amp --seeds 1 2 3 4 5
    ```

*   **ACM**:
    ```bash
    python main.py --epoch 200 --dataset ACM --n-fp-layers 2 --n-task-layers 1 --num-hops 4 --num-label-hops 4 --label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --amp --seeds 1 2 3 4 5
    ```

*   **IMDB**:
    ```bash
    python main.py --epoch 200 --dataset IMDB --n-fp-layers 2 --n-task-layers 4 --num-hops 4 --num-label-hops 4 --label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0. --amp --seeds 1 2 3 4 5
    ```

*   **Freebase**:
    ```bash
    python main.py --epoch 200 --dataset Freebase --n-fp-layers 2 --n-task-layers 4 --num-hops 2 --num-label-hops 3 --label-feats --residual --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --lr 3e-5 --weight-decay 3e-5 --batch-size 256 --amp --patience 30 --seeds 1 2 3 4 5
    ```

### OGBN-MAG 数据集运行 (进入 `ogbn/` 目录)

```bash
cd ogbn
```

*   **基础训练 (无额外嵌入)**:
    ```bash
    python main.py --stages 300 300 300 300 --num-hops 2 --label-feats --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --amp --seeds 1
    ```

*   **进阶训练 (使用 ComplEx 嵌入)**:
    1.  **生成 ComplEx 嵌入** (需要 `dgl < 1.0`):
        ```bash
        cd ../data/complex_nars
        python convert_to_triplets.py --dataset mag
        python embed_train.py --model ComplEx --batch_size 1000 --neg_sample_size 200 --hidden_dim 256 --gamma 10 --lr 0.1 --max_step 500000 --log_interval 10000 -adv --gpu 0 --regularization_coef 2e-6 --data_path . --data_files ./train_triplets_mag --format raw_udd_hrt --dataset mag
        # 假设保存路径为 ckpts/ComplEx_mag_0/mag_ComplEx_entity.npy
        python split_node_emb.py --dataset mag --emb-file ckpts/ComplEx_mag_0/mag_ComplEx_entity.npy
        cd ../../ogbn
        ```
    2.  **运行模型**:
        ```bash
        python main.py --stages 300 300 300 300 --extra-embedding complex --num-hops 2 --label-feats --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --amp --seeds 1
        ```

## 4. 注意事项
*   **显存优化**: 使用 `--amp` 参数开启自动混合精度训练。
*   **多种子运行**: 使用 `--seeds 1 2 3 4 5` 可以运行多次并取平均值。
*   **数据路径**: 确保 `data/` 目录结构正确，否则脚本可能无法找到数据集。
