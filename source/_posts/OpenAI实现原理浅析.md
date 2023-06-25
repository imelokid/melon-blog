---
title: OpenAI实现原理浅析
date: 2023-06-23 23:33:48
tags: [OpenAI, ChatGPT, "人工智能"]
---

# 机器学习的本质
让机器可以像人一样学习和思考，然后可以自动化的完成一些任务。这其中对应两个方面，输入(学习和理解)、输出(生产，完成任务)。在人类世界中，语言是信息交流的基础，是知识学习和传递的载体。因此，怎么才能让机器可以更好的理解和使用语言呢？这就是自然语言处理（NLP）的核心问题。

## 人是怎么学习和思考的？
==我叫圆圆==，我们是如何获取这句话的信息呢？要理解这句话的含义，我们要做到以下几点：
1. 知道句子中每个词的含义
2. 知道特定语言的语法，如本例中我是名词做主语，爱是动词做谓语，学习是名词做宾语。并且主谓宾是有一定的语序的。
3. 知道句子的语义，在知道了语法规则的情况下，可以基于上下文推断出句子的含义。
当然了，上面这句话比较简短。如果我们要理解一篇文章，或则一个很长的段落，我们还得学会从大量词语中抓住重点，提取关键信息。


## 机器是怎么学习和思考的？
从仿生学的角度上看，机器可以模仿人类的学习和思考的过程。对与给定的输入，通过一系列操作，得到预期的输出。这个过程的难点是：
1. 机器怎么可以像人一样理解语言
2. 机器如何基于理解到的信息给出预期的输出


# 文本语言特征化
我们以 !!我叫圆圆!! 为例，模拟人类理解语言的过程来看看机器应该怎么做？
机器并不能直接理解语言的含义，它只能处理0101这样的二进制数据，二进制有能比较方便的转换为十进制等其他进制的数据。机器的特长是对海量数据进行高速的运算。因此，我们需要想办法将句子中的词转换为数字，确切的说，转换为特定的向量。这个过程为词向量化。  
具体怎么做呢？还是要从我们自身取经。比如，我们有个朋友叫张三，那么我们应该来描述张三呢？我们可以说张三是个男生，张三是个程序员，张三喜欢打篮球，张三喜欢看电影。这样，我们就可以通过一系列的描述来描述张三。这些描述就是张三的特征，我们可以将这些特征组合成一个向量，这个向量就是张三的特征向量。同理，一段话中的每个词也具有一些特征，如词性，词频，在句子中的位置等。我们可以将这些特征组合成一个向量，这个向量就是这个词的特征向量。  
如何提取词的特征，目前有很多方法，比如one-hot、word2vec、glove等。这些方法的本质都是将词转换为向量，只是提取特征的方法不同而已。

> 词向量化的过程本质上是点坐标在不同坐标系中的坐标转换。为了保留更多的特征值，我们可以将点坐标转换到更高维的坐标系中。而为了降低计算量，我们可能还要将点坐标转换到更低维的坐标系中。
> 升维和降维的过程，我们可以利用现成的数学工具。如线性变换，非线性变换，矩阵变换等。


# One-hot
最简单的词向量化方法是one-hot，我们可以将每个词转换成一个向量，向量的维度为词典的大小，向量中只有一个元素为1，其他元素都为0。这个元素的位置就是这个词在词典中的位置。比如，我们有个词典，里面有三个词，分别是我、叫、圆圆。那么，我们可以将这三个词转换成如下的向量：

|  | 我 | 叫 | 圆圆 |
| --- | --- | --- | --- |
| 我 | 1 | 0 | 0 |
| 叫 | 0 | 1 | 0 |
| 圆圆 | 0 | 0 | 1 |

one-hot的缺点是向量的维度和词典的大小一致，当词典很大的时候，向量的维度也会很大，这样会导致计算量很大。而且，one-hot并不能很好的表达词与词之间的关系，比如，我和叫的关系是什么呢？我们可以说我是叫的主语，叫是我的谓语。但是，one-hot并不能很好的表达这种关系。

# Word Embedding
由于独热表示无法解决词之间相似性问题，这种表示很快就被词向量表示给替代了，这个时候聪明的你可能想到了在神经网络语言模型中出现的一个词向量 $C(w_i)$，对的，这个$C(w_i)$ 其实就是单词对应的 Word Embedding 值，也就是我们这节的核心——词向量。
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/NLP/NNLM-01.jpg" style="zoom:80%" />

NNLM是一种基于神经网络的语言模型，它的输入是一个词的one-hot向量，输出是这个词的概率。NNLM的结构如上图所示，它的输入层是一个one-hot向量，这个向量的维度为词典的大小。输入层的输出经过一个线性变换，然后经过一个非线性变换，最后经过一个线性变换，得到输出层的输出。输出层的输出是一个向量，向量的维度为词典的大小，向量中的每个元素代表这个词是这个词典中的哪个词的概率。

上图所示的神经网络语言模型分为三层，接下来我们详细讲解这三层的作用：
1. 神经网络语言模型的第一层，为输入层。首先将前 𝑛−1个单词用 Onehot 编码（例如：0001000）作为原始单词输入，之后乘以一个随机初始化的矩阵 Q 后获得词向量 $𝐶(𝑤𝑖)$，对这 𝑛−1个词向量处理后得到输入 $𝑥$，记作  $x=(C(w_1),C(w_2),\cdots,C(w_n))$
2. 神经网络语言模型的第二层，为隐层，包含 $h$ 个隐变量，𝐻代表权重矩阵，因此隐层的输出为 $𝐻𝑥+𝑑$，其中 𝑑为偏置项。并且在此之后使用$𝑡𝑎𝑛ℎ$作为激活函数。
3. 神经网络语言模型的第三层，为输出层，一共有 $|𝑉|$个输出节点（字典大小），直观上讲，每个输出节点 $y_i$是词典中每一个单词概率值。最终得到的计算公式为：
$y=𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑏+𝑊_𝑥+𝑈tanh(𝑑+𝐻𝑥))$，其中 𝑊是直接从输入层到输出层的权重矩阵，𝑈是隐层到输出层的参数矩阵。

上述过程比较复杂，简单来说，可以分为以下几步(拿我叫圆圆这段话举例)：
1. 获取我叫圆圆的One-hot编码，得到向量 $w = (w_1, w_2, \cdots, w_n)$。对于上述案例
    $w_1=[1,0,0]$
    $w_2=[0,1,0]$
    $w_3=[0,0,1]$
2. 接下来，通过输入层，将每个词转化为词向量，得到 $x=(C(w_1),C(w_2),\cdots,C(w_n))$，其中 $C(w_i)$ 为 $w_i$ 对应的词向量。
    $C(w_1)= w_1 . Q$
    $C(w_2)= w_2 . Q$
    $C(w_3)= w_3 . Q$
3. 然后，将 $C$ 作为隐层的输入，得到隐层的输出 $h=Utanh(d+Hx)$，其中 $d$ 为偏置项，$H$ 为权重矩阵。
4. 使用softmax函数对隐层的输出进行处理。
> softmax函数是一种常用的激活函数，其作用是将一个K维的向量压缩到另一个K维的向量，使得每个元素都在0和1之间，并且所有元素的和为1

$$
\begin{bmatrix}
0 & 0 & 0 & 1 & 0
\end{bmatrix} 
\begin{pmatrix}
17 & 23 & 1 \\
23 & 5 & 7 \\
4 & 6 & 13 \\
10 & 12 & 19 \\
11 & 18 & 25
\end{pmatrix} = 
\begin{bmatrix}
10 & 12 & 19
\end{bmatrix}
$$
但NNLM模型仍然存在一系列问题：
一个问题是，由于NNLM模型使用的是全连接神经网络，因此只能处理定长的序列。    
另一个问题是，由于其巨大的参数空间，将NNLM的训练太慢了。即便是在百万量级的数据集上，即便是借助了40个CPU进行训练，NNLM也需要耗时数周才能给出一个稍微靠谱的解来。显然，对于现在动辄上千万甚至上亿的真实语料库，训练一个NNLM模型几乎是一个impossible mission。 

通过对One-hot编码的词进行纬度转换后，我们可以将词分布到特定纬度的『超空间』中。基于这个『超空间』，我们可以很容易的计算出两个词之间的相似度。例如，我们可以计算出『我』和『你』之间的相似度，或者『我』和『苹果』之间的相似度。下图给了网上找的几个例子，可以看出有些例子效果还是很不错的，一个单词表达成 Word Embedding 后，很容易找出语义相近的其它词汇。
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/NLP/word-embedding-01.jpg" />
论文：https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

# WORD2REC
针对NNLM的问题，大神Mikolov注意到，原始的NNLM模型的训练其实可以拆分成两个步骤：
1. 用一个简单模型训练出连续的词向量；
2. 基于词向量的表达，训练一个连续的Ngram神经网络模型。  
而NNLM模型的计算瓶颈主要是在第二步。如果我们只是想得到word的词向量，是不是可以对第二步里的神经网络模型进行简化呢？就这样，他在2013年一口气推出了两篇paper，并开源了一款计算词向量的工具——至此，word2vec横空出世，主角闪亮登场。其中Word2Rec有两种模型：CBOW和Skip-gram。

## CBOW（Continues Bag-of-Words Model）
CBOW核心思想是从一个句子里面把一个词抠掉，用这个词的上文和下文去预测被抠掉的这个词；模型结构如下：
<img src="https://img2018.cnblogs.com/blog/1816627/201909/1816627-20190927110908257-516031113.png" />

## Skip-gram
CBoW模型依然是从context对target word的预测中学习到词向量的表达。反过来，我们能否从target word对context的预测中学习到word vector呢？答案显然是可以的：这便是Skip-gram模型
这个模型被称为Skip-gram模型，其模型结构与CBoW模型大同小异，也包括输入层、投影层（其实是多余的，加上该层以便与与CBoW模型对比）和输出层：

## word2vec的局限性
总的来说，word2vec通过嵌入一个线性的投影矩阵（projection matrix），将原始的one-hot向量映射为一个稠密的连续向量，并通过一个语言模型的任务去学习这个向量的权重，而这个过程可以看作是无监督或称为自监督的，其词向量的训练结果与语料库是紧密相关的，因此通常不同的应用场景需要用该场景下的语料库去训练词向量才能在下游任务中获得最好的效果。这一思想后来被广泛应用于包括word2vec在内的各种NLP模型中，从此之后不单单是词向量，我们也有了句向量、文档向量，从Word Embedding走向了World Embedding的新时代。word2vec非常经典，但也有其明显的局限性，其主要在以下几个方面：
1. 在模型训练的过程中仅仅考虑context中的局部语料，没有考虑到全局信息；
2. 对于英文语料，对于什么是词，怎样分词并不是问题（但个词就是独立的个体）。而对于中文而言，我们在训练词向量之前首先要解决分词的问题，而分词的效果在很多情况下将会严重影响词向量的质量（如分词粒度等），因此，从某些方面来说word2vec对中文不是那么的友好；
3. 在2018年以前，对于word2vec及其一系列其他的词向量模型都有一个相同的特点：其embedding矩阵在训练完成后便已经是固定了的，这样我们可以轻易从网上获取到大量预训练好的词向量并快速应用到我们自己任务中。但从另一个角度来说，对于同一个词，在任意一个句子，任意一个环境下的词向量都是固定的，这对于一些歧义词来说是存在较大问题的，这也是限制类似word2vec、Glove等词向量工具性能的一个很重要的问题。
比如Apple这个单词，在 I eat a Apple和 I use Apple这两个句子中，其实它的意思是不一样的。同样的Apple他有可能是真的🍎，也有可能指代📱。


## RNN和LSTM
这里穿插介绍下RNN（Recurrent Neural Network） 和 LSTM（Long Short-Term Memory）       
因为接下来要介绍的 ELMo（Embeddings from Language Models） 模型在训练过程中使用了双向长短期记忆网络（Bi-LSTM）。


## ELMO 






# Attention

# Transformer

# GPT


# 计算机如何理解自然语言
“the cat sat on the mat”，这句话对于人类来说是很容易理解的，但是对于计算机来说，这句话就是一堆字符，计算机并不知道这些字符代表什么意思。计算机只能理解数字，所以我们需要将自然语言转换成数字，这个过程就是自然语言处理（NLP）。
那么，我们来反思下人类是如何理解这句话的？
我们首先要要理解这句话中的每一个词，然后要理解对应语言的语法规则。对于每个词的含义，这个比较简单，我们通过死记硬背可以很快掌握。但是语法是什么呢？所谓语法就是对词按照一定的规则进行排序。这里面中的重点是每个词在句子中的位置信息。因此，单词含义+单词位置可以完整的表达这句话的含义。这就是我们人类理解自然语言的方式。
我们首先会将这句话分词，然后将每个词转换成数字，最后将这些数字组合成一个句子。计算机也是这样做的，但是计算机是如何将词转换成数字的呢？这就需要用到词嵌入（Word Embedding）技术了。
NLP就是如何在尽可能少的丢失信息的情况下，将自然语言转换成数字，然后再将数字转换成机器可以理解的语言。这个过程就是NLP的核心。

## 词袋技术实现原理
词袋技术是最简单的NLP技术，它的原理是将每个词转换成一个数字，然后将这些数字组合成一个向量。这个向量就是词袋。词袋的每个维度代表一个词，维度的值代表这个词在句子中出现的次数。这样，我们就可以将一句话转换成一个向量了。
词袋技术的缺点是没有考虑词的顺序，因此，它不能完整的表达句子的含义。比如，"the cat sat on the mat"和"the mat sat on the cat"这两句话的词袋是一样的，但是它们的含义是完全不同的。因此，词袋技术不能很好的表达句子的含义。
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd 

# 定义一个玩具语料库
toy_corpus = ["the fat cat sat on the mat", "this big cat slept", "the dog chased a cat"]

# 创建一个TfidfVectorizer对象
vectorizer = TfidfVectorizer()

# 使用vectorizer对象对toy_corpus进行训练，并返回文档-术语矩阵
corpus_tfidf = vectorizer.fit_transform(toy_corpus)

# 打印词汇表的大小
print(f"The vocabulary size is {len(vectorizer.vocabulary_.keys())}")

# 打印文档-术语矩阵的形状
print(f"The document-term matrix shape is {corpus_tfidf.shape}")

# 将稀疏矩阵转换为numpy数组，然后进行四舍五入，并将其作为pd.DataFrame的输入
df = pd.DataFrame(np.round(corpus_tfidf.toarray(), 2))

# 为数据框中的列命名，使得每个特征名称对应其TF-IDF值
df.columns = vectorizer.get_feature_names_out();

# 打印文档术语矩阵
print(df);
```
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/NLP/%E8%AF%8D%E8%A2%8B-1.jpg?q-sign-algorithm=sha1&q-ak=AKID-8mRK6N_FdhyGakU9d7yv79fEX2jed--QhLjoVV4-ZRqnOj6X_wTjjswJKDTmZIx&q-sign-time=1687540352;1687543952&q-key-time=1687540352;1687543952&q-header-list=host&q-url-param-list=ci-process&q-signature=f74ca9a75b1ccb7d87d7fa6992eb9af13f3aebd2&x-cos-security-token=4hnS2tQFOWPYLx3xEZOr7zS0Fs62v9lad9e01c1d70f25530a7bd6c5b30d22c66dJGAzbiMroA993u76-K0vWDSpbCuVWBee2CcdtNBMqJLw5VyZvPsi8wY-Ace_upK7ilba-IL4fRsAy4qa8SSedqD4OXAkwpYy6Tut3n-FBNpBSdMl0FQ3iYP1Q-qSBG-5b8NA3cWYUOYqtD9oq0mUNKycQFIlHoLarGiEvxZ3NGU5aJiqcDBuy81Skct2mecKsztOGf9IrTV99RhtvL02Q&ci-process=originImage" />

### 词袋的优缺点
|优点|缺点|
|:---|:---|
|1. 易于实现 <br/> 2.结果易于理解<br/> 3.适用于应用领域|1.会出现维度灾难<br/> 2.无法解决不可见单词的问题<br/> 3.难以捕捉语义关系，如属于(is-a)关系、包含(has-a)关系、同义词等<br/> 4.忽略词序信息<br/> 5.对于大型词汇表，速度较慢|


### 传统语言建模与生成方式
传统的语言建模与生成方式是基于n-gram的，它的原理是将句子中的每个词都看成是独立的，然后计算每个词出现的概率。这样，我们就可以通过计算每个词出现的概率来计算整个句子出现的概率了。这个过程就是语言建模。语言建模的目的是计算句子出现的概率，因此，我们可以通过计算句子出现的概率来判断这句话是否合理。如果句子出现的概率很低，那么这句话就是不合理的。这种建模方式可以分为三种：uni-gram、bi-gram和n-gram。这三种建模方式的区别在于考虑的词的个数不同。   
uni-gram只考虑一个词，用于估算单词在一个词汇表中的概率，即简单的计算该单词出现的频率与单词总数的比率。    
bi-gram又叫做一阶马尔可夫链，它用于估算一个词在给定前一个词的情况下出现的概率。P(wi|wi-1)。
n-gram多阶马尔可夫链，它用于估算一个词在给定前n-1个词的情况下出现的概率。P(wi|wi-1,wi-2,...,wi-n+1)。


# 使用深度学习
## 词嵌入
词嵌入是一种将词转换成向量的技术，它的原理是将每个词转换成一个向量，然后将这些向量组合成一个矩阵。这个矩阵就是词嵌入矩阵。词嵌入矩阵的每一行代表一个词，每一列代表一个维度。词嵌入矩阵的维度是一个超参数，我们可以根据需要来调整词嵌入矩阵的维度。词嵌入矩阵的维度越高，它能够表达的信息就越多，但是它的计算量也就越大。因此，我们需要根据实际情况来调整词嵌入矩阵的维度。
word2vec是一种流行的词嵌入技术，word2Vec在各种语法和语义语言任务都优于其他模型。
下面我们使用《麦克白》来训练一个word2vec模型，并进行文本相似度可视化。
```python
import nltk  # Natural Language Toolkit，一个NLP库
from nltk.corpus import gutenberg  # 导入NLTK的gutenberg语料库
from gensim.models import Word2Vec  # gensim的Word2Vec模型
import matplotlib.pyplot as plt  # 用于绘图的库
from sklearn.decomposition import PCA  # 用于降维的PCA算法
import random  # 用于生成随机数
import numpy as np  # 用于数值计算的库

nltk.download('gutenberg')  # 下载gutenberg语料库
macbeth = gutenberg.sents('shakespeare-macbeth.txt')  # 读取麦克白的句子
model = Word2Vec(sentences=macbeth, vector_size=100, window=4, min_count=10, workers=4, epochs=10)  # 创建Word2Vec模型
model.wv.similar_by_word('then',10)  # 找出与'然后'最相似的10个词

np.random.seed(42)  # 设置随机种子以获得可复现的结果
words=list([e for e in model.wv.key_to_index if len(e)>4])  # 选取长度大于4的词
random.shuffle(words)  # 随机打乱词的顺序
words3d = PCA(n_components=3, random_state=42).fit_transform(model.wv[words[:100]])  # 使用PCA降维

def plotWords3D(vecs, words, title):
    """
        Parameters
        ----------
        vecs : numpy-array
            Transformed 3D array either by PCA or other techniques
        words: a list of word
            the word list to be mapped
        title: str
            The title of plot     
    """
    fig = plt.figure(figsize=(14,10))  # 创建一个新的图形
    ax = fig.gca(projection='3d')  # 获取当前的axes，使用3D投影
    for w, vec in zip(words, vecs):  # 对于每个单词和对应的向量
        ax.text(vec[0],vec[1],vec[2], w, color=np.random.rand(3,))  # 在3D图上标出该点
    ax.set_xlim(min(vecs[:,0]), max(vecs[:,0]))  # 设置x轴的范围
    ax.set_ylim(min(vecs[:,1]), max(vecs[:,1]))  # 设置y轴的范围
    ax.set_zlim(min(vecs[:,2]), max(vecs[:,2]))  # 设置z轴的范围
    ax.set_xlabel('DIM-1')  # 设置x轴的标签
    ax.set_ylabel('DIM-2')  # 设置y轴的标签
    ax.set_zlabel('DIM-3')  # 设置z轴的标签
    plt.title(title)  # 设置图的标题
    plt.show()  # 显示图形

plotWords3D(words3d, words, "Visualizing Word2Vec Word Embeddings using PCA")  # 画出3D图
```
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/NLP/embedding-1.png?q-sign-algorithm=sha1&q-ak=AKIDbfJYYMn5DhyzrPmkhatjUlUszybDK7eCGkBxUt7CMdMGt1AvQfzLZy-4oLgh8P9p&q-sign-time=1687540274;1687543874&q-key-time=1687540274;1687543874&q-header-list=host&q-url-param-list=ci-process&q-signature=70ab2f6e8d0c15b28f296306c63ef06f5bbb9d3d&x-cos-security-token=3e6mJMONs0ou42d4wVJCI6h4kiTFNUxac161d7573c28fe874e55f4d19a4f3caaofnqwYFtVjyum0B4dkyvPf1wat3xp4NtUgy5FljyyASTyVKI_QTRjZKwB8Uf5bEuo_APzr7P6CqyLTwmtAijaExoiKrGeqG_KiJZSnOHavmpQXG6_bU_NeBCqUZSr9Wey_UK7V5oWajZlVJzQkia17Hx70NFiKNki0PrsN_3zOlxw1eJHS_hdgx-ywsuC2RHs7RodEh0DnwmSsvKSHqEOg&ci-process=originImage" />


