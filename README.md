
**任务明确**：

- 重写模型，设计窗卷积核，先将三个特征统一映射到 (u, d) ，在不经过 transformer 的情况下直接分类，跑实验结果。
- 加上transformer encoder
- 提取双模态融合信息
- 提取三模态融合信息
- 使用全局注意力机制进行特征融合**任务明确**：

### 窗卷积

作用：

- 把 $X^{in} \in R^{S \times E}$ 映射到统一空间 $H^{in} \in $R^{t \times d}$
- 其中，t = 


#### A

$X^{in}_A \in R^{220 \times 73}$ 

#### V

$X^{in}_V \in R^{350 \times 512}$ 

#### L

$X^{in}_L \in R^{610 \times 200}$ 
