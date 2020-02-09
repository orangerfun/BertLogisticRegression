# 1.数据集
使用的数据集为[SST2](https://nlp.stanford.edu/sentiment/index.html)，该数据集收集了一些影评中的句子，每个句子都有标注，好评被标注为 1，差评标注为 0。

# 2. 目标
创建一个分类器，它的输入是一句话（即类似于数据集中的句子），并输出一个 1（表示这句话体现出了积极的情感）或是 0（表示这句话体现出了消极的情感）。整个模型由两个子模型组成：
* DistilBERT 先对句子进行处理，并将它提取到的信息传给下个模型。DistilBERT 是 BERT 的缩小版，它是由 HuggingFace 的一个团队开发并开源的。它是一种更轻量级并且运行速度更快的模型，同时基本达到了 BERT 原有的性能。
* 另外一个模型，是 scikit learn 中的 Logistic 回归模型，它会接收 DistilBERT 处理后的结果，并将句子分类为积极或消极（0 或 1）。

尽管整个模型包含了两个子模型，但是我们只需要训练 logistic 回归模型。至于 DistillBERT，我们会直接使用已经在英文上预训练好的模型。然而，这个模型不需要被训练，也不需要微调，就可以进行句子分类。BERT 训练时的一般目标让我们已经有一定的句子分类能力了，尤其是 BERT 对于第一个位置的输出（也就是与 [CLS] 对应的输出）。我觉得这主要得益于 BERT 的第二个训练目标——次句预测（Next sentence classification）。该目标似乎使得它第一个位置的输出封装了句子级的信息。Transformers 库给我们提供了 DistilBERT 的一种实现以及预训练好的模型。

# 3. 环境
* torch
* transformers(pytorch transformer)
* python 3.x

# 4. 

