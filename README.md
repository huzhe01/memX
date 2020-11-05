## Memory-GAN
Memory based GAN for few-shot image generation

## Data Preparation
1.Omniglot:https://drive.google.com/drive/folders/15x2C11OrNeKLMzBDHrv8NPOwyre6H3O5  

2.VGGFace:https://drive.google.com/drive/folders/15x2C11OrNeKLMzBDHrv8NPOwyre6H3O5  

3.Animals: 

## Enviroment:
以下几个需要安装：
imageio              2.9.0  

opencv-python        4.4.0.44  

Pillow               8.0.1  

tensorboard          1.13.1  

tensorflow-gpu       1.13.1  

## 主要代码文件说明:
- 训练部分
0. 数据准备：/data_with_matchingclassifier.py  
1. ./dagan_architectures_with_matchingclassifier.py  
dagan_architecutre：重新设计了DAGAN的结构，主要包括Uresnet_gernerator和Discriminator。
2. ./dagan_networks_wgan_with_matchingclassifier.py
写了DAGAN这个类，继承了Uresnet和Discriminator。并包含了计算loss的方法。
3. ./experiment_builder_with_matchingclassifier.py
这个是入口函数，GAN网络结构的初始化，写了run_experiment方法。
4. ./train_dagan_with_matchingclassifier.py 
准备data,实例化expriment_builder,调用run_experiment方法
 
- 生成部分  

1.generation_builder_with_matchingclassifier.py
生成图片的实验设置
2.test_dagan_with_matchingclassifier_for_generation.py
run_generate.sh: 入口函数，准备data,实例化expriment_builder,调用run_experiment方法

目前的网络结构使用的是U-net一种encoder+decoder网络的结构，参考了DAGAN(https://github.com/AntreasAntoniou/DAGAN)
## todo
代码主要是基于DAGAN这篇文章做的改动，优先使用VGGface数据集，实现下面任务。训练可以设置为1way-3shot的形式。
### 目前需要实现功能: 
- [ ] 记忆力机制, 如何读取和存储
- [ ] 高斯混合采样
- [ ] 其他多样性生成的探索(如插值)


### Note:以下任务在testing data上做
### 小样本生成图像效果对比(重点对比):

- [x] FIGR

- [x] DAGAN

- [x] MATCHING GAN

- [x] F2GAN

  PS: 
  
- [x] GAN的评价指标实现IS,FID(已经实现，还未测试效果)

  完成状态：代码跑通，但尚未系统性地整理结果

### 分类任务: 
low data classification: 拿一个训练好的分类器做分类任务

- [x] 传统数据增强方法
- [ ] memory GAN
- [x] DAGAN
- [x] Matching GAN
- [x] F2GAN
- [ ] GMN
- [ ] DAWSON

few-shot classification: 暂时hold

### 增量学习任务:
- [ ] Memory GAN
- [ ] DAGAN
- [ ] Matching GAN
- [ ] F2GAN

### ablation study：

- [ ] 给定的图片数量
- [ ] 有无记忆力机制
- [ ] 有无混合高斯分布

