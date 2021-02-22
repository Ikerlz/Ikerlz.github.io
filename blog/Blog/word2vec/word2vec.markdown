## Word2vec [center]

> [TOC]

### 一、语言模型

#### 1、语言模型的概念

语言模型是计算一个句子是一个句子的概率的模型，一个句子既需要符合语法规则，同时也需要符合语义，只有两方面都满足的情况下，语言模型才会给出一个较高的概率值。语言模型是一个无监督的模型，不需要任何的语料标注。


#### 2、语言模型的发展

##### （1）基于专家知识的语言模型

所谓的基于专家知识的语言模型就是语言学家企图总结出一套通用的语法规则，例如形容词后接名词，但是这种方法并无法当今网络用语频繁出现的这样一种情况，因此现在不用常用这种方法

##### （2）统计语言模型

所谓的统计语言模型是指通过概率计算来刻画语言模型

$$
\begin{aligned}
P(s)=& P\left(w_{1}, w_{2}, \dots, w_{n}\right) \\
=&P\left(w_{1}\right) P\left(w_{2} | w_{1}\right) P\left(w_{3} | w_{1} w_{2}\right) \dots P\left(w_{n} | w_{1} w_{2} \dots w_{n-1}\right)
\end{aligned}
$$

- 某一个词语出现概率的求解方法：用语料的频率代替概率（频率学派）
$$
p(w_i) = \frac{count(w_i)}{N}
$$

- 条件概率如$P(w_2 | w_1)$的求解方法：频率学派+条件概率

$$
p\left(w_{i} | w_{i-1}\right)=\frac{p\left(w_{i-1}, w_{i}\right)}{p\left(w_{i-1}\right)} = \frac{count(w_{i-1},w_i)}{count(w_{i-1})}
$$

通常概率表达的方式，我们就能构造出一个较好的语言模型，但仍然存在下面两个问题：
- 由于$P(s)$采用的是连乘方式，因此对于某一个词语$w_i$，如果在语料中没有出现过，那么$P(w_i)=0$，这样会导致$P(s)=0$
- 如果一句话由非常多的短语构成，同时某些短语在语料中出现的次数较少，那么多个小于1的概率相乘必然会使得$P(s)$趋向于零

如何对上面两个问题进行解决呢？$\Rightarrow$ **平滑操作**



##### （3）统计语言模型中的平滑操作

- 有一些词或词组在语料中没有出现过，但是这不能代表它不可能存在，而平滑操作就是给那些没有出现过的词或者词组一个不为零的较小的概率。最常用的就是 *Laplace Smoothing* ，也称之为**加1平滑**，具体操作为：每个词在原来出现的次数上加1
- 例如一个语料库中，$A$出现的次数为0，$B$为990，$C$为10，则$A$、$B$、$C$的概率为0、$0.99$、$0.01$；如果通过加1平滑，则相当于$A$出现的次数为1，$B$为991，$C$为11，则$A$、$B$、$C$的概率为0.001、$0.988$、$0.011$。从上面的例子就可以看出，进行加1平滑后，出现次数多的词语的概率计算值有所下降，而出现次数少的词语概率计算值有所上升，加1平滑本质上起到了一个“劫富济贫”的作用
- 加1平滑只要词有显著的作用，对于很长的词组短语并没有很好的效果，这是因为这种长的短语很稀疏，如果要平滑操作，则所有的短语都要平滑，这是因为**参数空间太大**，**数据稀疏**严重，如何解决这两个问题呢？需要用到**马尔可夫假设**

##### （4）马尔可夫假设

在统计语言模型中的马尔可夫假设主要内容就是：下一个词出现的概率仅依赖于前面的一个词或几个词，如果与前$k$个词有关，则称为$k-gram$模型

$$
\begin{aligned}
unigram \qquad P(s) = & P(w_1)P(w_2)\ldots P(w_n) \\
bigram \qquad P(s) = & P(w_1)P(w_2 | w_1)\ldots P(w_n | w_{n-1}) \\
trigram \qquad P(s) = & P(w_1)P(w_2 | w_1)P(w_3 | w_2w_1)\ldots P(w_n | w_{n-1}w_{n-2}) \\
k-gram \qquad P(s) = & P(w_1)P(w_2 | w_1)P(w_3 | w_2w_1)\ldots P(w_n | w_{n-k+1}\ldots w_{n-1}) \\
\end{aligned}
$$

注意：$k-gram$模型的参数空间为$V+V^2+\ldots+V^k$，$V$为语料库词语总数


#### 3、语言模型的评价指标：困惑度（Perplexity）

语言模型本质上是一个多分类问题，这里可以通过困惑度（Perplexity）进行模型的评价：

$$
PP(s) = P(w_1, w_2, \ldots, w_n)^{-\frac{1}{n}}=\sqrt[n]{\frac{1}{P\left(w_{1}, w_{2}, \dots, w_{n}\right)}} = \sqrt[n]{\frac{1}{P(s)}}
$$


因此，句子概率越大，语言模型越好，困惑度越小

### 二、词的表示方法

#### 1、独热表示（One-hot Representation）

所谓的独热表示，就是一个向量，只有一个地方为1，其他位置都为0。这种表示方法的优点是：表示简单，但是缺点也十分明显，那就是当词非常多的时候，表示向量的维度非常高，需要的空间非常大，同时还有一个问题就是这种表示方法无法表示词与词之间的关系

#### 2、分布式表示方法（Distributed Representation）

分布式表示也称为稠密表示，对于每一个词，仅需要一个$D$维的向量就能表示，这里的$D$是远远小于总的词语个数$V$，对于独热表示，每一个都需要一个$V$维的向量，而分布式表示相当于是把$V$维压缩到了$D$维，但$D$维向量中某一位置的数值大小需要通过训练得到。同时，分布式表示有一个很大的好处，那就是可以表示词与词之间的相似性（利用余弦相似度）—— *Word embedding*


### 三、论文学习

#### 1、论文背景知识

1986年， *Hinton* 第一次提出了 *Distributed Representation* ，目的是将离散形式的词输入到神经网络中，由于神经网络需要的是连续的值，因此提出了一种 *Distributed Representation* 的表示方法，把离散形式的词进行连续的分布式表示；2003年，在文章 **A Neural Probabilistic Language Model**首次使用了词向量，而在2003年到2013年间，提出了很多训练词向量的方法，但共同的缺点就是非常慢，无法在大的预料上训练，在2013年提出的 *word2vec* 模型解决了训练速度慢的问题，能够在大的语料上进行训练，得到更好的词向量，推动了自然语言处理的发展


#### 2、论文的研究成果

- 提出了一种新的模型结构
- 提出优化训练的方法，使得训练速度加快
- 废除训练代码 *word2vec* ，使得单机训练成为可能
- 成果：训练的词向量又快又好，并且能够在大规模预料上进行词向量的训练

#### 3、论文的研究意义

##### （1）衡量词向量之间的相似程度

对于两个词的相似度，可以用公式：

$$sim(word1, word2) = \cos(wordvec1, wordvec2)$$

进行计算，计算得到的值越接近于1，则两个词越相似，一种更客观的评价方式是**词类比（analogy）**，计算公式为

$$\cos(wordvec1-wordvec2+wordvec3,wordvec4)$$

通过这种方式，能够学习到词与词之间的关系，例如 *France* 和 *Paris* 与 *Italy* 和 *Rome* 这两个词对就是非常相似的，因此$V(France) - V(Paris) \approx V(Italy) - V(Rome)$，这种模型的训练就能得到非常多在语义上相似的词对


##### （2）作为预训练模型提升NLP任务

可以将训练得到的词向量可以用于外部任务比如命名实体识别、文本分类；也可以应用于其他NLP任务上，相当于一个半监督任务，可以有效地提高模型的泛化能力


### 四、论文精读

#### 1、对比模型

对比模型主要对比两种模型，一种是**前馈神经网络语言模型（NNLM）**，一种是**循环神经网络模型（RNNLM）**


##### （1）前馈神经网络语言模型（NNLM）

前馈神经网络语言模型最早由 *Bengio* 在2003年提出，网络结构如下图所示：

![](assets/截屏2020-06-2821.29.07.png?r=60)[center]

从上图中可以看出，NNLM的结构分为三层，最下面为输入层： *input layer* ，输入的是词的索引而非词本身， 输入后，将每个 *index* 映射为一个向量，最后将这些向量 *contact* 在一起；中间一层为隐藏层： *hidden layer* ，即将输入层的向量接上一个全连接层，用 $tanh$作为激活函数激活；最上面为输出层： *output layer*，也是接上一个全连接层，利用$softmax$输出概率。这里输入的是前$n-1$个词语，最后输出的是第$n$个位置单词的概率。下面详细介绍三个层：
- 输入层：将词映射为向量，相当于一个$1\times V$的 *one-hot* 向量乘以一个 $V\times D$的向量得到一个 $1\times D$ 的向量
- 隐藏层：一个以$tanh$为激活函数的全连接层：$a=tanh(d+Ux)$
- 输出层：一个全连接层，后面接个$softmax$函数来生成概率分布：$y=b+Wa$，其中$y$是一个$1\times V$的向量，即：
$$
P\left(w_{t} \mid w_{t-n+1}, \ldots, w_{t-1}\right)=\frac{\exp \left(y_{w_{t}}\right)}{\sum_{i} \exp \left(y_{i}\right)}
$$

##### Remark：语言模型困惑度和Loss的关系
对于一句话，$Loss$的定义和困惑度的定义如下：
- $Loss$：$L=-\frac{1}{T} \sum_{i=1}^{T} \log p\left(w_{i} \mid w_{i-n+1}, \dots, w_{i-1}\right)$
- 困惑度：$PP(s)=P\left(w_{1}, w_{2}, \ldots, w_{T}\right)^{-\frac{1}{T}}=\sqrt[T]{\frac{1}{P\left(w_{1}, w_{2}, \ldots, w_{T}\right)}}$
对困惑度取对数得到$\log(PP(s)) = -\frac{1}{T}\log P\left(w_{1}, w_{2}, \ldots, w_{T}\right)$，根据全连接公式和马尔可夫假设可以得到$\log(PP(s))= -\frac{1}{T} \sum_{i=1}^{T} \log p\left(w_{i} \mid w_{i-n+1}, \dots, w_{i-1}\right) = e^{Loss}$，所以困惑度是自然对数的$Loss$次方，这就是两者的关系


**讨论：** *Bingio* 在2003年的文章中提出的$NNLM$是一篇开山之作，也为后面的研究提出了几个研究的思路，具体如下：
- 仅对一部分输出进行梯度传播：像$the$，$a$这类词语，出现次数多，但是在模型中并没特别重要作用，因此对于这一类词语，可以不进行梯度传播
- 引入先验知识，如词性等：2003年这篇模型并没有输入词性，因此提出了词性的输入是否可以提高模型的精度这个问题，其中主要有两个子问题：
 - 网络模型自身能否学习到词性
 - 网络模型如果能自己学习词性，学到的词性知识是否够用
- 解决一词多义问题，对于同一词语的不同意思，模型如何进行区分
- 加速$softmax$层，由于输出层是全连接层，对每一个词语都需要输出概率，因此非常慢，需要进行加速

##### （2）循环神经网络语言模型（RNNLM）

![](assets/截屏2020-06-2822.30.20.png?r=40)[center]
如上图所示，$RNNLM$也有三层：

- 输入层：和$NNLM$一样，需要将当 前时间步的转化为词向量（ *one-hot* ）
- 隐藏层：对输入和上一个时间步的隐藏输出进行全连接层操作：
$$
s(t)=U w(t)+W s(t-1)+d
$$
- 输出层：一个全连接层，后面接个$softmax$函数来生成概率分布，$y(t) = b + Vs(t)$，其中，$y$是一个$1\times V$的向量，
$$
P\left(w_{t} \mid w_{t-n+1}, \ldots, w_{t-1}\right)=\frac{\exp \left(y_{w_{t}}\right)}{\sum_{i} \exp \left(y_{i}\right)}
$$
循环神经网络语言模型并没有使用马尔可夫假设，因为每个时间步预测一个词，在预测第$n$ 个词时，已经使用了前$n-1$个词的信息


#### 2、Word2vec

##### （1）Log-linear model
定义：将语言模型的建立看成个多分类问题，相当于线性分类器加上$softmax$，这样就构成了一个$Log-$线性模型：$Y = softmax(Wx+b)$，多分类的逻辑回归模型就是一个$Log-$线性模型，$word2vec$中的两种模型：$skip-gram$和$CBOW$都是$Log-$线性模型

##### （2）Word2vec 原理
- 语言模型基本思想：句子中下一个词的出现和前面的词是有关系的，所以可以使用前面的词预测下一个词
- $word2vec$基本思想：句子中**相近的词**之间是有联系的，比如今天后面经常出现上午，下午和晚上。所以$word2vec$的基本思想就是用词来预测词：
 - $skip-gram$使用中心词预测周围词
 - $CBOW$使用周围词预测中心词
- $word2vec$可以看成是对语言模型的一种简化

##### （3）Skip-gram

$Skip-gram$是用中心词预测周围词，若中心词为$w_i$，这里需要定义在中心词附近多大的范围才是中心词，既需要一个$window$，例如我们取$window=2$，则我们需要计算的就是：$P(w_{i-1} | w_i), P(w_{i-2} | w_i), P(w_{i+1} | w_i), P(w_{i+2} | w_i)$，那么$Skip-gram$是如何求取这些概率值的呢？前面讲到，求这种概率的问题本质上是一个多分类问题，以$P(w_{i-1}| w_i)$为例，输入的是$w_i$，$label$是$w_{i-1}$，计算过程如下图所示：
![](assets/截屏2020-06-2910.39.08.png?r=60)[center]

注意：
- 输入的是$w_i$的索引
- $W$为中心词矩阵，$W \in \mathbb{R}^{V\times D}$；$W^{\prime}$为周围词矩阵，$W^{\prime} \in \mathbb{R}^{D\times V}$
- 最后通过$softmax$，求得每个位置的一个概率，为一个$1\times V$的向量，我们需要第$i-1$个位置的值最大，因此通过不断的反向传播，训练$W^{\prime}$和$W$
- 计算公式可写为：
$$
p\left(w_{i-1} \mid w_{i}\right)=\frac{\exp \left(u_{w_{i-1}}^{T} v_{w_{i}}\right)}{\sum_{w=1}^{V} \exp \left(u_{w}^{T} v_{w_{i}}\right)}
$$
 ------
$Skip-gram$的损失函数：

$$
J(\theta)=\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \leq j \leq m, j \neq 0} \log p\left(w_{t+j} \mid w_{t}\right)
$$

需要对$J(\theta)$求最大值，或者对$J(\theta)$取相反数并求取最小值

##### （4）CBOW

![](assets/截屏2020-06-2910.57.50.png?r=50)[center]

$CBOW$的结构如上图所示，是通过周围词预测中心词，$BOW$指的是$Bag \ of \ word$，即**词袋模型**，因此$CBOW$是基于词袋模型的一种模型，具体细节如下图所示：

![](assets/截屏2020-06-2911.00.32.png?r=60)[center]

从上面可以看出，$CBOW$也是使用了$W$和$W^{\prime}$，只是中间进行了一个加和，最后的优化过程仍然是优化$W$和$W^{\prime}$

- - - - - 
损失函数：

$$
J(\theta)=\frac{1}{T} \sum_{T} \sum \log P(c \mid o)=\frac{1}{T} \sum \frac{\exp \left\{u_{o}^{T} v_{c}\right\}}{\sum_{j=1}^{V} \exp \left\{u_{o}^{T} v_{j}\right\}}
$$
其中,
$$
P(c \mid o)=\frac{\exp \left\{u_{o}^{T} v_{c}\right\}}{\sum_{j=1}^{V} \exp \left\{u_{o}^{T} v_{j}\right\}}
$$
$u_0$是窗口内上下文词向量之和，$v_c$、$v_j$是中心词向量‘


##### （5）关键技术

$word2vec$中，最后一层需要输出$V$个概率，因此需要一些降低复杂度的方法：
- 层次$softmax$：*Hierarchical Softmax*
- 负采样：*Negative Sampling*
- 高频词的重采样：*Subsampling of Frequent Words*

**下面分别介绍这三种方法**

- - - - - 

###### 1）Hierarchical Softmax

层次$softmax$的基本思想是将求$softmax$操作转化为求$sigmoid$操作，对于$softmax$，我们需要做$V$次操作，而如何转化为$sigmoid$，则可以采用**树结构**，这样就只需要$\log_2^V$次操作。如下图所示，如果是一个8字符的情况（$a \sim h$），则需要对每个字符做$softmax$，而如果将其构建为下图所示的**满二叉树**的结构，则对于任意字符，仅需要3次就能找到，即$\log_2^V$次


![](assets/IMG_6F9AA8F61ED6-1.jpeg?r=60)[center]


能否找到比$\log_2^V$还要快的结构呢？答案是肯定的，通过构建 *Huffman* 树，找到带权重路径最短的二叉树，可以进一步加速，层次$softmax$就是通过这种方式进行构建，具体的构建方法（$Skip-gram$模型）如下图所示：
![](assets/截屏2020-06-2915.11.56.png?r=60)[center]

- 每一个分支节点（即存在$child$的节点）都是一个向量：$\theta_0$、$\theta_1$等
- 以单词$I$为例，计算公式为：
$$
\begin{aligned}
p(\mathrm{I} \mid c) &=\sigma\left(\theta_{0}^{T} v_{c}\right) \sigma\left(\theta_{1}^{T} v_{c}\right)\left(1-\sigma\left(\theta_{2}^{T} v_{c}\right)\right) \\
&=\sigma\left(\theta_{0}^{T} v_{c}\right) \sigma\left(\theta_{1}^{T} v_{c}\right) \sigma\left(-\theta_{2}^{T} v_{c}\right) \qquad \qquad \sigma(x) = \frac{1}{1+ e^{-x}}
\end{aligned}
$$
- 这里的$\theta$参数就相当于上下文词向量，大概有$\log V$个，而树的高度为：$L(w) = O(\log_2^V)$
- - - - - 
**公式表达**：
$$
p\left(w \mid w_{I}\right)=\prod_{j=1}^{L(w)-1} \sigma \left([\![ n(w, j+1)=\operatorname{ch}(n(w, j))]\!] \cdot {v_{n(w, j)}^{\prime}}^{\top} v_{w_{I}}\right)
$$
- $n(w,j)$表示词$w$在第$j$个节点上，$n(w,1)$表示 *root* 节点，$n(w,L(w))$表示叶子节点
- $\operatorname{ch}(n(w,j))$表示节点$n(w,j)$的 *right child node*
- 双中括号表示，如果括号中为 *true* ，则为1，否则为-1，即：
$$
[\![ x]\!]=\left\{\begin{array}{ll}
1, & \text { if } x \text { is true } \\
-1, & \text { else }
\end{array}\right.
$$

- $v_{W_{I}}$为中心词的词向量，${v_{n(w, j)}^{\prime}}$为词$w$在树上的第$j$个节点的参数
- - - - - -

**CBOW中的层次$softmax$：**
前面提到的层次$softmax$是基于$Skip-gram$模型进行构建的，而针对$CBOW$模型的构建方法如下图所示：
![](assets/截屏2020-06-2923.32.50.png?r=50)[center]

可以看到，$CBOW$中的层次$softmax$构建方式与$Skip-gram$相似，唯一 的区别在于$u_0$，这里的$u_0$是值**窗口内上下文词向量的平均**

- - - - - 
###### 2）Negative Sampling
- - - - - 
**核心思想：**
负采样的核心思想是：**舍弃多分类**以提高速度，在现在的研究中，层次$softmax$并不常用，因为本质上仍然是一个多分类问题，而负采样在现在的研究中应用十分广泛，原因就在于直接舍弃了多分类，将其变为了一个二分类问题。例如有一个训练样本，中心词是$w$，它周围上下文共有$2c$个词，记为$context(w)$。由于这个中心词$w$的确和$context(w)$相关存在，因此它是一个真实的正例。通过负采样，我们得到$neg$个和$w$不同的中心词$w_i,i=1,2,\ldots,neg$，这样$context(w)$和$w_i$就组成了$neg$个并不真实存在的负例。利用这一个正例和$neg$个负例，我们进行二元逻辑回归，得到负采样对应每个词$w_i$对应的模型参数$\theta_i$，和每个词的词向量
- - - - - 
**损失函数：**
$$
J_{\operatorname{neg}-\operatorname{sample}}(\theta)=\log \sigma\left(u_{o}^{T} v_{c}\right)+\sum_{k=1}^{K} E_{k \sim P(w)}\left[\log \sigma\left(-u_{k}^{T} v_{c}\right)\right]
$$
- $v_c$是中心词向量
- $u_0$是窗口内上下文词向量
- $u_k$是负采样上下文词向量

- - - - - 
**采样方法：**

$$P(w)=\frac{U(w)^{\frac{3}{4}}}{Z}$$

- $U(w)$是词$w$在数据集中出现的频率，$Z$为归一化的参数，使得求解之后的概率和依旧为1
- 例如$U(a)=0.01, U(b)=0.99$，则
    - $P(a)=\frac{0.01 \times 0.75}{0.01 \times 0.75+0.99 \times 0.75}=0.03$
    - $P(b)=\frac{0.99 \times 0.75}{0.01 \times 0.75+0.99 \times 0.75}=0.97$

- 通过这样的采样方法，可以减少频率大的词的抽样概率，增加评率小的词的抽样概率

- - - - - 
**CBOW中的负采样**
$CBOW$中的负采样与$Skip-gram$中的负采样形式相似，损失函数为：
$$
J(\theta)=\log \sigma\left(u_{o}^{T} v_{c}\right)+\sum_{i=1}^{k} E_{j \sim P(w)}\left[\log \sigma\left(-u_{o}^{T} v_{j}\right)\right]
$$
- $u_0$是窗口内上下文词向量的平均
- $v_c$是正确的中心词向量
- $v_j$是错误的中心词向量

- - - - - 


###### 3）Subsampling of Frequent Words

自然语言处理有一个共识，即文档或者数据集中出现频率高的词往往携带信息较少，比如 *the, is, a, and*等，而出现频率低的词往往携带信息多。因此应当重点关注这些低频词，这也是重采样的目的，而进行**重采样**的原因有以下两个：

- 想更多地训练重要的词对，比如训练 *France* 和 *Paris* 之间的关系比训练 *France* 和 *the* 之间的关系要有用
- 高频词很快就训练好了，而低频次需要更多的轮次
- - - - - 
**重采样的方法**

$$
P\left(w_{i}\right)=1-\sqrt{\frac{t}{f\left(w_{i}\right)}}
$$

$f(w)$为词$w_i$在数据集中出现的频率，$t$为一个超参数，文中$t$选取为$10^{-5}$，训练集中的词$w_i$会以$P(w_i)$的概率被删除。词频越大，$f(w_i)$越大，$P(w_i)$越大，那么词$w_i$就有更大的概率被删除，反之亦然。如果词$w_i$的词频小于等于$t$，那么$w_i$则不会被别除，这样就加速训练，能够得到更好的词向量。

- - - - - 
##### （6）小结
$word2vec$可以总结为“$2+3$”或者“$2+2+1$”，第一个“$2$”表示两种模型，一种是$Skip-gram$模型，是用中心词预测周围词，一种是$CBOW$模型，是用周围词预测中心词”“$3$“表示的是三种关键技术：层次$softmax$、负采样、重采样；也可以分为“$2+1$”，本质上是一样的

#### 3、模型复杂度

##### （1）模型复杂度的概念
- - - - - 
模型复杂度用$O$表示，代表的是训练的复杂度，计算公式为：

$$
O=E \times T \times Q
$$

- $O$是训练复杂度（ *training complexity* ）
- $E$是迭代次数（ *number of training epochs* ）
- $T$是数据集大小（ *number of words in the training set* ）
- $Q$是模型计算复杂度（ *model computational complexity* ）

对不同的模型进行复杂度比较时，需要保证$E$和$T$相同，因此实际上是比较$Q$的大小，而$Q$的计算可以通过计算模型的参数个数来等价代替，从而进行模型之间复杂度的比较
- - - - - 

##### （2）NNLM复杂度
![](assets/截屏2020-06-3014.39.27.png?r=40)[center]
$NNLM$是输入$N-1$个词，预测第$N$个词，每个输入的词都会被映射为一个$D$维的向量，因此输入层的参数$x$的维度为$N\times D$，隐藏层$W$是一个全连接层，因此$W$的维度为$N \times D \times H$，输出层$U$也是一个全连接层，因此$U$的维度为$H\times V$，因此$NNLM$的复杂度$Q$为：
$$
Q=V \times H+N \times D \times  H+N \times D
$$
如果使用层次$softmax$，则复杂度可降为$Q=\log_2^V \times H+N \times D \times  H+N \times D$
- - - - - 

##### （3）RNNLM复杂度
![](assets/截屏2020-06-3014.45.23.png?r=40)[center]

$NNLM$输入的是$w(t)$，维度$1\times D$，$U$为全连接层，维度为$D \times H$，$W$为$H \times H$，输出层$V$为$H \times V$，因此

$$Q = 1\times D + D\times H + H \times H + H \times V$$
由于$D \approx H$，因此$Q = H(2H +1)+H\times V$，因此可以写为：

$$Q = H \times H + H \times V$$

如果使用层次$softmax$，则$Q = H \times H + H \times \log_2^V$

- - - - - 
##### （4）CBOW模型复杂度
![](assets/截屏2020-06-3014.55.06.png?r=40)[center]
如果周围词个数为$N$，则输入层的维度为$N\times D$，输出层为$D\times V$，因此
$$Q = N\times D + D\times V$$

- 如果使用层次$softmax$，则$Q=N\times D+D \times \log_2^V$
- 如果使用负采样，则$Q=N\times D+D \times (K+1)$

- - - - - 
##### （5）Skip-gram模型复杂度
![](assets/截屏2020-06-3014.50.59.png?r=40)[center]
- - - - - 

对于一个中心词，维度为$D$，周围词矩阵$W^{*}$为$D\times V$，同时需要求解$C$个周围词，因此$Q$为：
$$
Q=C(D+D \times V)
$$

- 如果使用层次$softmax$，则$Q=C(D+D \times \log_2 V)$
- 如果使用负采样，则$Q=C(D+D \times (K+1))$
- - - - - 

##### （6）复杂度比较

不同模型的复杂度$Q$如下：
NNLM：$Q = N \times D+N \times D \times H+H \times \log _{2} V$
RNNLM：$Q= H \times H+H \times \log _{2} V$
CBOW + HS：$Q=N \times D+D \times \log _{2} V$
Skip-gram + HS：$Q=C\left(D+D \times \log _{2} V\right)$
CBOW + NEG：$Q=N \times D+D \times(K+1))$
Skip-gram + NEG：$Q=C(D+D \times (K+1))$

- $word2vec$中的两种模型时间复杂度都比$NNLM$和$RNNLM$要低
- $Skip-gram$比$CBOW$要稍微慢一些
- 负采样一般比层次$softmax$要快一些

- - - - - 

### 五、论文总结与启发

- - - - - 
#### 1、关键点
- 更简单的预测模型：$word2vec$
- 更快的分类方案：$HS$和$NEG$
- - - - - 

#### 2、创新点
- 使用词对的预测来替代语言模型的预测
- 使用$HS$和$NEG$降低分类复杂度
- 使用$subsampling$加快训练
- 新的词对推理数据集来评估词向量的质量

- - - - - 


#### 3、启发点
- 大数据集上的简单模型往往强于小数据集上的复杂模型
- *King* 的词向量减去 *Man* 的词向量加上 *Woman* 的词向量和 *Queen* 的词向量最接近
- 我们决定设计简单的模型来训练词向量，虽然简单的模型无法像神经网络那么准确地表示数据，但是可以在更多地数据上更快地训练
- 我们相信在更大的数据集上使用更大的词向量维度能够训练得到更好的词向量
- - - - - 

### 六、代码实现


实现$Skip-gram$和$CBOW$以及$HS$和$NGE$，具体代码见我的[Github]()