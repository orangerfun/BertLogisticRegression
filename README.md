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

# 4. 流程
(1) 将句子分词（包括subword）并进行编码（索引）和padding
(2) 将输入向量传递给 DistilBert 之后的工作方式就跟在 BERT 中一样，每个输入的词都会得到一个由 768 个浮点数组成的输出向量。<br>
![](https://github.com/orangerfun/BertLogisticRegression/raw/master/DistilBert.png)<br>
由于这是个句子分类任务，我们只关心第一个向量（与 [CLS] 对应的向量）。该向量就是我们输入给 logistic 回归模型的向量。<br>
![](https://github.com/orangerfun/BertLogisticRegression/raw/master/all.png)<br>

# 5.参考
* https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb
* [Bert快速入门指南](https://mp.weixin.qq.com/s?__biz=MzU3NjE4NjQ4MA==&mid=2247486729&idx=2&sn=4e607e4d82fce53807b760b1192b5be0&chksm=fd16fc16ca617500337d464c9374705dfdcf8a522e990c07a05096886f34c09b724c78b77ead&scene=0&xtrack=1&key=956aab8526dddd2f6f6857470f07376315d7ba18fc98b1b2d276786d9d1b4da87fe5705f6d6132c9aa98298e7752d2bd64551332e5fdd2bc3565e4db836f1dd106ae6707047a041e3e56bf0f036d6182&ascene=14&uin=MjUwMzAyMzA2Mg%3D%3D&devicetype=Windows+10&version=62070158&lang=zh_CN&exportkey=Aa%2Fv46F0haL%2BEBCuyAkPwpo%3D&pass_ticket=qtOC5Cpe9%2FWtRHIdqFXRmywHx0MTQN7K58ewsCedWiwtYA55Y1hOunyT45%2FgOEtO)


