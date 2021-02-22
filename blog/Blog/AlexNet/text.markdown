## AlexNet[center]
[TOC]
AlexNet最早是由 *Alex Krizhevsky* 等人于2012年提出的，论文标题为[《Imagenet classification with deep convolutional neural networks》](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

### 一、论文学习


#### 1、论文简介
本篇本文提出了采用了关键技术，例如$ReLU$激活函数，$Dropout$等技术，同时使用了双GPU进行模型的训练，这些技术在当下的深度学习模型中也经常使用，因此这篇文章具有非常高的学习价值


#### 2、论文重点

##### （1）ReLU激活函数
一般的模型训练使用的是**饱和非线性**激活函数，例如$f(x)=tanh(x)=(1+e^{-x})^{-1}$，本文使用的是$ReLU$激活函数，即：$f(x)=\max(0,x)$，是一种**不饱和**的激活函数，主要有以下三个优点：
- 使网络训练更快
- 防止梯度消失(弥散)
- 使网络具有稀疏性
![](assets/截屏2020-06-2222.58.58.png?r=50)[center]

上图为论文中的插图，从上图可以看出，使用$ReLU$激活函数（实线）比$tanh$激活函数训练速度更快

##### （2）多GPU训练

本文采用了双GPU对模型进行并行训练，提高了模型训练的速度。双GPU或多GPU训练都会涉及到不同GPU数据的交流问题，本文中的这个交流问题在介绍AlexNet结构时着重讲解


##### （3）Local Response Normaliza（LRN）

- LRN翻译为局部响应标准化，有助于$AlexNet$泛化能力的提升，是收到了真实神经元**侧抑制现象**(lateral inhibition)启发而提出的一种方法，所谓的**侧抑制**，是一种生物学上的概念，指的是细胞分化变为不同时，它会对周围细胞产生抑制信号，阻止它们向相同方向分化，最终表现为细胞命运的不同
- $LRN$公式：
$$
b_{x, y}^{i}=a_{x, y}^{i} /\left(k+\alpha \sum_{j=\max (0, i-n / 2)}^{\min (N-1, i+n / 2)}\left(a_{x, y}^{j}\right)^{2}\right)^{\beta}
$$
- 公式参数介绍：
 - $a^i$：代表第$i$个神经元的激活值
 - $b^i$：代表第$i$个神经元通过LRN操作后的激活值
 - $k$：超参数，由原型中的bias指定
 - $\alpha$：超参数，由原型中的alpha指定
 - $\beta$：超参数，由原型中的beta指定
 - $n/2$：超参数，由原型中的deepth_radius指定
 - $x,y$：像素的位置，公式中用不到
 - $i,j$：代表通道 *channel*

- 示意图：
![](assets/截屏2020-06-2223.58.24.png?r=59)[center]

从上图中可以看出，$x,y$分别代表的是 *width* 和 *height* ，而$i,j$代表的是 *channel* 的索引，这里假设的 *channel* 索引为0到$N-1$，因此可以看出公式中的求和符号$\Sigma$的上下限求最大和最小的目的就是为了防止索引超过 *channel* 的范围，而$n/2$表示只有在距离第$i$个神经元$n/2$范围内的神经元才会对第$i$个神经元有抑制作用

- 局限性：在本文的数据集上，通过$LRN$方法，使得模型准确率提高了1%，但是2014年的一篇文章证明了$LRN$并不是一种通用的方法，是无效的，而且现阶段有更好的正则化方法，例如 *Batch Normalization* 等，因此 *LRN* 方法逐渐被淘汰


##### （4）Overlapping Pooling

如果定义$z$为 *kernel size* 的大小，$s$为步长大小（ *stride* ），则通常情况下是$s=z$，这种情况下是没有重叠的，但是如果$s<z$，就会出现重叠的情况，我们称这种情况为 *Overlapping Pooling* ，论文中是采用了$s=2,z=3$的池化层，提升了模型精度。

##### （5）数据增强方法（Data Augmentation）
数据增强能够有效地避免过拟合，论文中主要针对图片的**位置**和**色彩**进行数据增强，文章通过这两个方面的数据增强，能够从一张图片得到了2048张图片。

- 方法一：针对位置
 - 训练阶段：
          1. 图片统一缩放至$256\times256$
          2. 随机位置裁剪出$224\times224$区域
          3. 随机进行水平翻转
 - 测试阶段：
         1. 图片统一缩放至$256\times256$
         2. 裁剪出5个$224\times224$区域
         3. 均进行水平翻转，共得到10张$224\times224$图片
- 方法二：针对色彩
 - 通过PCA方法修改RGB通道的像素值，实现颜色扰动，但是这种方法效果有限

几点说明：
- 从上面对位置的操作可以得到为什么能够从一张图片得到了2048张图片，由于裁剪过程是随机选择，因此有$(256 - 244)^2 =1024$种可能，同时又进行水平翻转，因此再乘2，得到2048
- 训练阶段的裁剪是随机裁剪，并且论文中提到，是需要先将短边裁剪到224像素，然后再在长边中心裁剪得到$224\times224$；在测试阶段的裁剪并不是随机的，而是在左上、左下、右上、右下以及中心这五个位置进行裁剪，然后翻转得到10张图片，并且这10张图片都要输入到模型中得到的概率值再取平均
- 对色彩的处理提升效果并不明显，并且涉及到矩阵的分解（PCA），因此现阶段很少用到这种方法对图像色彩进行扰动

##### （6）Dropout

Dropout技术是一个非常实用的减轻过拟合现象的一种技术，在现阶段的深度学习中也是一种非常常用的技术

![](assets/截屏2020-06-2301.43.52.png?r=60)[center]

如上图所示，左边为不使用Dropout的情况，下一层的某一神经元与上一层的每个神经元都保持连接，而右边是使用了Dropout的情况，可以看出下一层的某一神经元只与上一层的某些神经元保持连接，具体与哪些神经元连接并不是固定不变的，这就体现出了**随机**的效果，通常会设置 *dropout probability* 来实现**“随机”**，一般设置为0.5，这里需要注意的是，Dropout是用于训练过程中，因此在测试过程中，为了保证数据尺度的一致性，我们必须对测试过程中的神经元输出值乘以 *dropout probability* ，这一点是非常关键的

#### 3、AlexNet结构


- 论文中提到AlexNet包含有八个带权值的层，其中有5个是卷基层，有3个是全连接层，这仅仅是带权值的层，如果加上不带权值的层，例如池化层，就不止8个层了
- 在5个卷积层中，第2、4、5个卷积层均只与在同一GPU上的前一层进行连接，而 第3层是对双GPU上的所有前一层的神经元都进行了连接，完成了不同GPU上信息的交流
- LRN的位置在第一和第二个卷积层中间，而池化层的位置在第1、2、5卷积层后面，ReLU用于所有的卷积层和全连接层

![](assets/截屏2020-06-2313.38.50.png?r=80)[center]

- 从上图中可以看出，整个网络的结构分为上下两层，表示两个GPU上的训练（注意，上下两层应该是完全一样的，但是原文的图片上层是不完整的，理论上应该是与下层结构完全一致），输入的是一个$224\times224\times3 = 150528$维的向量，输出的是一个1000维的向量，代表的是1000类的分类结果
- 从上图中可以看出，图中并没有包含到ReLU、LRN等操作，只有卷积层和全连接层，每一层后的后续操作如下：
 -  $Conv1\  \rightarrow \ ReLU \  \rightarrow \ Pool \  \rightarrow \ LRN$
 -  $Conv2\  \rightarrow \ ReLU \  \rightarrow \ Pool \  \rightarrow \ LRN$
 -  $Conv3\  \rightarrow \ ReLU $
 -  $Conv4\  \rightarrow \ ReLU $
 -  $Conv5\  \rightarrow \ ReLU \  \rightarrow \ Pool$
![](assets/截屏2020-06-2313.56.07.png?r=80)[center]
- AlexNet中，每个阶段特征图的变化情况如下图所示，这里与论文不同的是输入的像素为$227\times227$，其实在早期的AlexNet中，就是$227\times227$的输入，这对于结果是没有任何影响的，因为根据卷积输出特征图大小公式：$F_{o}=\left \lfloor \frac{F_{\mathrm{in}}-k_{s}+2 p}{s}\right \rfloor+1$，如果是$227\times227$，则$padding = 0$即$p=0$，因此输出的特征图大小为$\lfloor \frac{227-11}{4} \rfloor+1 = 55$，如果是$224\times224$，则需要加入$padding$，通常$padding=2$，因此输出的特征图大小为$\lfloor \frac{224-11+2\\times 2}{4} \rfloor+1 = 55$，因此两种情况下卷积后得到的特征图大小相同，都为$55\times55$，而通道数都是96，这是因为卷积核的个数为96个，因此通道数为96，后面的特征图大小的计算都可以通过计算得到。而论文中提到了AlexNet结构一共有六千万的参数，这里的计算方式是将每一层的参数个数计算出来再相加，而对某一层的参数个数的计算公式为：$count = F_{i} \times\left(K_{\mathrm{s}} \times K_{\mathrm{s}}\right) \times K_{n}+K_{n}$，其中$F_{i}$为通道数，$K_s$为卷积核的大小，$K_n$为卷积核的个数，最后加上的$K_n$称之为偏置（$bias$），以第一层为例，第一层的参数个数为$3\times(11\times11)\times96+96 = 34944$，每一层的参数个数计算如下图所示：

![](assets/截屏2020-06-2314.22.33.png?r=70)[center]

**从上图中可以看出，FC1这个全连接层的参数个数占到了总参数个数的一半以上，因此到了后期的神经网络模型，FC层使用率大大下降，因为全连接层会占据大量的内存**


#### 4、实验分析

##### （1）卷积可视化

![](assets/截屏2020-06-2314.45.34.png?r=60)[center]

论文将第一个卷积层后的特征图进行了可视化，一共是2个GPU上的96个卷积核，前3行为第一个GPU学习到的特征，主要学习到了图片的纹路等结构特征，而后3行为第二个GPU学习到的特征，主要学习到了图片色彩方面的特征，可以看出，两个GPU上的学习到的特征并不相同

##### （2）高级特征的相似性

论文实验发现，ALexNet提取到的高级特征之间具有很强的相关性，即**相似图片的第二个全连接层输出特征向量的欧式距离相近**，如下图所示，如果两张图片的第二个全连接层输出特征向量的欧式距离很近，则这两张图片应该是非常相似的，此外，论文中提到这种相似的图片在没有输入到模型时，计算两张图片的欧式距离并不是非常小的，这就启发我们可以使用$4096$的特征向量进行比较相似性，这就可以用来做**图像的检索**、**图像的编码**、**图像的聚类**，这样可以大大的减小复杂度。


![](assets/截屏2020-06-2314.51.46.png?r=60)[center]
#### 3、论文总结与启发

##### （1）关键点
 - 大量带标签数据——ImageNet
- 高性能计算资源——GPU
- 合理算法模型——深度卷积神经网络

##### （2）创新点
- 采用ReLu加快大型神经网络训练
- 采用LRN提升大型网络泛化能力
- 采用Overlapping Pooling提升指标
- 采用随机裁剪翻转及色彩扰动增加数据多样性
- 采用Drpout减轻过拟合

##### （3）启发点
- 深度与宽度可决定网络能力
- 更强大GPU及更多数据可进一步提高模型性能
- 图片缩放细节，对短边先缩放
- ReLU不需要对输入进行标准化来防止饱和现象，即说明sigmoid/tanh激活函数有必要对输入进行标准化

### 二、代码实现

代码实现采用[猫狗数据集](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)，使用`Pytorch`实现，主要有下面5个部分


#### 1、构建`DataLoader`
分别构建训练集和测试集对应的`DataLoader`：
```python
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=4)
```
这里需要构建一个`Dataset`类——`CatDogDataset`，并使用下面两行代码构建MyDataset实例：
```python
train_data = CatDogDataset(data_dir=data_dir, mode="train", transform=train_transform)
valid_data = CatDogDataset(data_dir=data_dir, mode="valid", transform=valid_transform)
```
其中`CatDogDataset`类代码实现如下：

```python
class CatDogDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=620, transform=None):
        """
        猫狗分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        self.split_n = split_n
        self.data_info = self._get_img_info()  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def _get_img_info(self):

        img_names = os.listdir(self.data_dir)
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

        random.seed(self.rng_seed)
        random.shuffle(img_names)

        img_labels = [0 if n.startswith('cat') else 1 for n in img_names]

        split_idx = int(len(img_labels) * self.split_n)  # 25000* 0.9 = 22500
        # split_idx = int(100 * self.split_n)
        if self.mode == "train":
            img_set = img_names[:split_idx]     # 数据集90%训练
            # img_set = img_names[:22500]     #  hard code 数据集90%训练
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别，仅支持(train, valid)")

        path_img_set = [os.path.join(self.data_dir, n) for n in img_set]
        data_info = [(n, l) for n, l in zip(path_img_set, label_set)]

        return data_info
```

在构建自定义的`Dataset`时，需要重写`__getitem__`和`__len__`这两个方法，`__len__`主要是记录数据的大小，而`__getitem__`是通过图片索引寻找图片，这里需要主要的是`__getitem__`是会不停的被调用，因此不能在`__getitem__`中进行过多的操作，否则会导致训练过程非常缓慢，对于图片信息的获取就不建议放到`__getitem__`中，这里是将图片信息的获取过程单独写成一个函数`_get_img_info`，通过这个函数得到所有图片的信息并存储到一个列表中，这样在`__getitem__`中就只需要传入这个列表对对列表中的元素就行索引，得到图片的信息，避免了重复操作，加快了代码的运行速度


#### 2、构建模型

Pytorch中已经实现了$AlexNet$，因此直接调用即可，在调用后需要加载预训练参数以加快模型训练速度：
```python
alexnet_model = get_model(path_state_dict, False)

def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model
```
另外，$AlexNet$是一个1000类的分类网络，而我们使用的数据集是二分类任务，因此需要替换最后的输出层：
```python
num_ftrs = alexnet_model.classifier._modules["6"].in_features
alexnet_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes) # num_classes = 2
```
#### 3、构建损失函数

由于是二分类问题，因此直接采用**交叉熵损失函数即可**：
```python
criterion = nn.CrossEntropyLoss()
```

#### 4、构建优化器
优化器选择传统的**随机梯度下降**即可：
```python
optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)
```
当然，也可以选择冻结卷积层：
```python
fc_params_id = list(map(id, alexnet_model.classifier.parameters()))  # 返回的是parameters的 内存地址
base_params = filter(lambda p: id(p) not in fc_params_id, alexnet_model.parameters())
optimizer = optim.SGD([
    {'params': base_params, 'lr': LR * 0.1},  # 0
    {'params': alexnet_model.classifier.parameters(), 'lr': LR}], momentum=0.9)
```

#### 5、迭代训练

```python
for epoch in range(start_epoch + 1, MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    alexnet_model.train()
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = alexnet_model(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().cpu().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        alexnet_model.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                bs, ncrops, c, h, w = inputs.size()     # [4, 10, 3, 224, 224
                outputs = alexnet_model(inputs.view(-1, c, h, w))
                outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                loss = criterion(outputs_avg, labels)

                _, predicted = torch.max(outputs_avg.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                loss_val += loss.item()

            loss_val_mean = loss_val/len(valid_loader)
            valid_curve.append(loss_val_mean)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_mean, correct_val / total_val))
        alexnet_model.train()
```

最后的训练损失曲线如下图所示：

![](assets/2020-06-23-17-03-43.jpg?r=60)[center]

从上图可以看出训练效果非常好，在测试集上的准确率能够达到97.52%，如果多训练几个Epoch，准确率会进一步提高