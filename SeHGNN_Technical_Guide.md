# SeHGNN 技术指南：异构图节点分类预训练与实践

**任务名称**：异构图节点分类 (Heterogeneous Graph Node Classification)
**项目性质**：预训练模型实践 (Pre-trained Model Practice) —— 通过预计算元路径特征与标签传播实现高效的异构图表示学习。

---

## 1. 训练流程 (Training Process)

### 1.1 预计算阶段 (Pre-computation)
在模型训练前，SeHGNN 通过预计算来捕捉图的结构信息。

**对应代码 (`hgb/main.py`)**:
```python
# 特征传播：沿着元路径聚合邻居特征
g = hg_propagate_feat_dgl(g, tgt_type, args.num_hops, max_length, extra_metapath, echo=True)

# 标签传播：利用训练集标签作为额外特征
meta_adjs = hg_propagate_sparse_pyg(adjs, tgt_type, args.num_label_hops, max_length, ...)
for k, v in meta_adjs.items():
    label_feats[k] = remove_diag(v) @ label_onehot
```

### 1.2 模型训练阶段
**对应代码 (`hgb/main.py`)**:
```python
for epoch in tqdm(range(args.epoch)):
    # 调用 train 函数进行单轮训练
    loss, acc = train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, scalar=scalar)
```

---

## 2. 数据传递 (Data Flow)

数据从原始图结构转化为多通道特征，最后输入模型。

**对应代码 (`hgb/model.py` - `forward`)**:
```python
def forward(self, batch, feature_dict, label_dict={}, mask=None):
    # 1. 将不同元路径的特征投影到统一维度
    x = [features[k] for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
    x = torch.stack(x, dim=1) # [Batch, Channels, Dimension]
    
    # 2. 特征投影与语义融合
    x = self.feature_projection(x)
    x = self.semantic_fusion(x).transpose(1, 2)
    
    # 3. 映射到下游任务
    x = self.fc_after_concat(x.reshape(B, -1))
    return self.task_mlp(x)
```

---

## 3. 下游任务应用 (Downstream Tasks)

本实践中的下游任务是节点分类。

**对应代码 (`hgb/model.py`)**:
```python
# 定义下游任务 MLP
self.task_mlp = nn.Sequential(
    nn.PReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden, hidden),
    nn.BatchNorm1d(hidden),
    nn.PReLU(),
    nn.Linear(hidden, nclass) # 输出维度为类别数
)
```

---

## 4. 训练结果展示 (Results)

### 4.1 控制台日志结果
训练时会输出每轮的准确率（Micro-F1 和 Macro-F1）。

**展示结果**:
```text
Epoch 45, training Time(s): 0.0286, estimated train loss 0.3093, acc 99.1786, 99.1629
evaluation Time: 0.0254, Train loss: 0.2706, Val loss: 0.3433, Test loss: 0.3593
Train acc: (97.74, 97.68), Val acc: (95.88, 95.57), Test acc: (95.35, 95.00)
```

---

## 5. 结果获取与导出 (Result Export)

训练完成后，模型会自动生成用于评估的预测文件。

**对应代码 (`hgb/main.py`)**:
```python
# 生成 HGB 官方评估格式的 txt 文件
dl.gen_file_for_evaluate(
    test_idx=test_nid, 
    label=pred, 
    file_name=f"{args.dataset}_{args.seed}_{checkpt_file.split('/')[-1]}.txt"
)

# 保存全图预测概率张量
torch.save(all_pred, f'{checkpt_file}.pt')
```

**最终产出**:
1.  **`*.txt`**: 包含测试集节点的预测标签，可直接提交至 HGB 榜单。
2.  **`*.pt`**: 包含所有节点的 Softmax 概率，可用于后续的可视化或集成学习。
