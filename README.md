# RotatEreplaceTransE


## 属性嵌入模块

合并KG1和KG2成一个KG，对应AT={AT1, AT2}

需要将实体嵌入一个统一的语义空间，在这个统一的空间中计算不同KG的实体间的距离，从而找到要对齐的实体。

使用TransE来嵌入属性三元组，使用参数共享来统一嵌入表示。

评分函数：A=PA+KA=参数共享模块分+属性嵌入模块分

### 参数共享模块

因为对齐的实体在不同KG中有相同含义，也就是有相同的嵌入表示。

方法是参数共享。

对每个对齐实体对(e1,e2)，定义e1===e2。这个模型中没有变量，所以其评分为PA=0.

这很简单，不会添加变量和变量漂移，也不会引入嵌入错误。

### 属性嵌入模块

使用TransE嵌入属性三元组AT到语义向量空间。

将三元组（e,a,v）的a视为e到v的平移。通过在嵌入空间中不断调整e,a,v，我们有 e+a ≈ v

energy函数E(e,a,v)=|| e+a-v ||

用基于margin的评分函数训练。

最终从TransE输出的嵌入中获得所有实体的属性嵌入AE

## 数据集说明

DBP15k(JAPE) 0.3

## RDGCN

Source code and datasets for IJCAI 2019 paper: ***[Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs](https://arxiv.org/pdf/1908.08210.pdf)***.

Initial datasets are from [GCN-Align](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE).

### Dependencies

* Python>=3.5
* Tensorflow>=1.8.0
* Scipy>=1.1.0
* Numpy

### Datasets

Please first download the datasets [here](http://59.108.48.35/data.tar.gz) and extract them into `data/` directory.

There are three cross-lingual datasets in this folder:
- fr-en
- ja-en
- zh-en

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);
* zh_vectorList.json: the input entity feature matrix initialized by word vectors;

### Running

* Modify language or some other settings in *include/Config.py*
* cd to the directory of *main.py*
* run *main.py*

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to wyting@pku.edu.cn.
