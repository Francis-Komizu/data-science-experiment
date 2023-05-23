### CPC

在对比损失中定义边界值 $m$ 的目的是为了控制同类样本之间的距离和不同类样本之间的距离之间的差异。如果没有边界值 $m$ 的限制，同类样本之间的距离可以无限制地变小，而不同类样本之间的距离可以无限制地变大，这可能会导致嵌入空间中的聚类效果不佳。

边界值 $m$ 可以看作是同类样本之间距离的最小阈值。如果同类样本之间的距离小于 $m$，则对比损失不会产生任何损失项，因此模型不会进一步优化将它们之间的距离缩小。相反，如果不同类样本之间的距离小于 $m$，则对比损失会产生正向损失项，鼓励模型将它们之间的距离进一步增加。

通常情况下，边界值 $m$ 可以根据具体任务和数据集进行选择。如果同类样本之间的距离较小，可以尝试增加 $m$，以便更好地区分同类样本和不同类样本。相反，如果同类样本之间的距离较大，可以尝试减小 $m$，以便更好地聚集同类样本。在实际应用中，可以通过交叉验证等方法来选择最佳的 $m$ 值。



### 对比损失

对比损失（Contrastive loss）是一种常用的深度学习损失函数，通常用于学习数据的嵌入表示（embedding representation）。对比损失的目标是将同类样本（positive samples）的嵌入距离拉近，将不同类样本（negative samples）的嵌入距离推远，使得嵌入空间中同类样本聚集在一起，不同类样本分散开来。

考虑一个简单的对比损失形式，假设我们有一对同类样本  $(x_{i}, x_{j})$ 和一对不同类样本 $(x_{i}, x_{k})$，其中 $x_i$ 是一个输入样本，$x_j$ 和 $x_k$ 分别是与 $x_i$ 属于同一类和不同类的样本。我们将 $x_{i}$ 的嵌入表示表示为 $f(x_{i})$，并使用一个相似度度量函数 $d$ 来计算 $f(x_{i})$ 和 $f(x_{j})$、$f(x_{k})$ 之间的距离。对于一个对比损失函数 $L_{c}$，可以定义为：
$$
L_{c} = \frac{1}{2N} \sum_{i=1}^{N} y_{i} d(f(x_{i}), f(x_{j})) + (1-y_{i}) \max(m - d(f(x_{i}), f(x_{k})), 0)
$$
其中 $N$ 是训练样本的数量，$y_{i}$ 是一个二元变量，表示 $x_{i}$ 和 $x_{j}$ 是否属于同一类，$m$ 是一个预先定义的边界值，$d$ 是一个相似度度量函数，如欧几里得距离或余弦相似度。

对于同类样本，我们希望它们的距离较近，因此对应的损失项为 $y_{i} d(f(x_{i}), f(x_{j}))$；对于不同类样本，我们希望它们的距离较远，因此对应的损失项为 $(1-y_{i}) \max(m - d(f(x_{i}), f(x_{k})), 0)$。其中 $\max(m - d(f(x_{i}), f(x_{k})), 0)$ 表示如果 $f(x_{i})$ 和 $f(x_{k})$ 之间的距离小于 $m$，则损失项为 $0$，否则为 $m-d(f(x_{i}), f(x_{k}))$。这个损失函数的目标是使同类样本之间的距离尽可能小，不同类样本之间的距离尽可能大，从而在嵌入空间中形成可分离的聚类。

在对比损失中定义边界值 $m$ 的目的是为了控制同类样本之间的距离和不同类样本之间的距离之间的差异。如果没有边界值 $m$ 的限制，同类样本之间的距离可以无限制地变小，而不同类样本之间的距离可以无限制地变大，这可能会导致嵌入空间中的聚类效果不佳。

边界值 $m$ 可以看作是同类样本之间距离的最小阈值。如果同类样本之间的距离小于 $m$，则对比损失不会产生任何损失项，因此模型不会进一步优化将它们之间的距离缩小。相反，如果不同类样本之间的距离小于 $m$，则对比损失会产生正向损失项，鼓励模型将它们之间的距离进一步增加。

通常情况下，边界值 $m$ 可以根据具体任务和数据集进行选择。如果同类样本之间的距离较小，可以尝试增加 $m$，以便更好地区分同类样本和不同类样本。相反，如果同类样本之间的距离较大，可以尝试减小 $m$，以便更好地聚集同类样本。在实际应用中，可以通过交叉验证等方法来选择最佳的 $m$ 值。




