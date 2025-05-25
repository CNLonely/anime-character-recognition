# anime-character-recognition
训练和识别动漫人物

本项目旨在实现一个高性能、可扩展的动漫人物人脸识别系统，适用于大规模数据场景。通过结合深度学习模型（ResNet-50）与增强策略（如 CutMix、MixUp、对抗训练等），本系统能够在复杂背景、相似角色间仍保持较高的识别准确率。

主要功能包括：

多角色分类模型训练（支持大规模人物库）

自动划分训练集/验证集及标签预处理

多尺度特征提取与归一化，用于后续检索/聚类任务

基于 YOLO 的动漫人脸自动检测与切割

支持图像输入的一键识别，适用于推理部署

# 展示
<img src="assets/1.png" alt="1" style="width:100%;"/>
<img src="assets/2.png" alt="2" style="width:100%;"/>


## 角色数据集
由于目前互联网上尚无高质量、结构规范的动漫人物识别数据集，本项目同时提供了整理好的角色图像数据集，可用于训练与测试。

角色集下载地址(目前已收集了1006名动漫人物，共82部动漫)：
链接: https://pan.baidu.com/s/19i4gsfgd8_Wr9Y7A_y1mtQ?pwd=k1x5 提取码: k1x5

### 数据来源

数据集来源于多个公开动漫图像资源，包括但不限于：

- 动漫作品截图
- 部分开源项目与角色爬虫脚本

数据已清洗，并按角色进行分类组织。

### 数据结构
项目使用的数据需按以下结构组织：

```项目结构
datasets/
  └── face_crop/
        ├── full/
        │     ├── id_00001/
        │     │      └── xxx.jpg
        │     ├── id_00002/
        │     │      └── xxx.jpg
        │     └── ...
        ├── train/
        │     ├── id_00001/
        │     │      └── xxx.jpg
        │     ├── id_00002/
        │     │      └── xxx.jpg
        │     └── ...
        └── val/
              ├── id_00001/
              ├── id_00002/
              └── ...
```
`meta/character_meta.json` 是一个包含角色元数据的 JSON 文件，用于映射角色 ID 到其对应的名称和出处。该文件对于训练、验证和推理阶段都至关重要。
```json
{
    "id_00001": {
        "name": "角色A",
        "anime": "动漫名字"
    },
    "id_00002": {
        "name": "角色B",
        "anime": "动漫名字"
    },
    ...
}
```
* 每个角色一个文件夹，文件夹名为角色 ID 或角色名称。

## 一、训练说明

本项目基于 ResNet-50 实现动漫角色识别，采用多种训练增强技术（CutMix、MixUp、对抗训练等）以提升模型鲁棒性和准确率。

###  环境建议

* 推荐使用带有 NVIDIA GPU 的环境运行，支持 CUDA。
* 若使用 GPU，将启用自动混合精度加速处理。

### 1.数据准备

项目使用的数据需按以下结构组织：

```项目结构
datasets/
  └── face_crop/
        ├── full/
        │     ├── id_00001/
        │     │      └── xxx.jpg
        │     ├── id_00002/
        │     │      └── xxx.jpg
        │     └── ...
        ├── train/
        │     ├── id_00001/
        │     │      └── xxx.jpg
        │     ├── id_00002/
        │     │      └── xxx.jpg
        │     └── ...
        └── val/
              ├── id_00001/
              ├── id_00002/
              └── ...
```

* 每个角色一个文件夹，文件夹名为角色 ID 或角色名称。
* 可以直接把数据集放在full中，然后运行/utils/dataset_split.py自动划分
* 训练集和验证集分别放在 `train/` 和 `val/` 文件夹中。
* 请确保存在角色信息文件：`meta/character_meta.json`，格式为包含角色名称与 ID 的映射。
---

### 2.环境依赖

建议使用 Python 3.8 及以上版本，依赖的主要包如下：

在使用本项目前，请先安装依赖
```pip
pip install -r requirements.txt
```

---

### 3.训练方法

运行主程序：

```python
python train_character_classification.py
```

默认配置如下：

* 模型结构：ResNet-50，使用 ImageNet 预训练参数
* 学习率：0.0003
* 优化器：AdamW
* 批量大小：64
* 训练轮数：100
* 标签平滑因子：0.1
* 数据增强：CutMix 和 MixUp（概率各为 0.5）
* 对抗训练：使用 FGSM（epsilon=0.01）
* 早停策略：验证准确率连续 5 次未提升则停止训练

---

### 4.模型输出与日志

训练过程中将输出以下信息：

* Top-1 / Top-5 准确率
* Macro Recall 值
* 每类角色的准确率（训练集和验证集）

保存文件如下：

* 最佳模型参数：`models/anime_character_recognition_best.pth`
* 各类别准确率：`models/best_epoch_class_accuracies.csv`

训练结束后还会绘制一张图表，显示准确率低于 70% 的角色（默认不弹出界面）。

---

### 5.模型结构说明

本项目使用 ResNet-50 作为基础模型结构，加载预训练参数后将输出层替换为适配当前分类数的全连接层：

```python
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

---

### 6.增强与正则化技术

本项目使用了以下训练策略以提升模型效果：

**数据增强：**

* RandAugment（rand-m5-n1）
* 随机仿射变换、颜色扰动、灰度变换
* 透视变换、随机噪声、随机擦除、模糊

**鲁棒性提升：**

* CutMix
* MixUp
* 标签平滑损失（Label Smoothing）
* FGSM 对抗训练

**采样策略：**

* 使用类别平衡采样器（ClassAwareSampler）提升小众角色识别能力

---

如需修改训练参数或模型结构，可直接编辑 `train.py` 文件中主函数部分，或自行添加命令行参数接口。

---

## 二、多尺度特征提取

当模型训练完毕后，请运行以下命令进行多尺度融合特征提取：

```
python .\utils\extract_embeddings.py
```

该脚本将使用训练好的模型（位于 `models/anime_character_recognition_best.pth`），提取图像的特征表示并保存，用于后续的人物检索或聚类分析。

### 1.功能说明

* 加载已训练模型，移除分类头，仅保留 ResNet50 的特征提取部分。
* 对每张图像进行三种尺度（224、256、288）处理，提取多尺度特征。
* 对不同尺度下的特征进行平均融合并归一化。
* 最终输出以下三个文件：

  * `meta/embeddings.npy`：图像的特征向量（二维数组）
  * `meta/labels.npy`：图像的类别标签（整数索引）
  * `meta/image_paths.txt`：图像路径列表（文本文件）

### 2/数据目录结构要求

图像数据应放在如下结构中：

```
datasets/
└── face_crop/
    └── train/
        ├── id_00001/
        │   ├── img1.jpg
        │   └── ...
        ├── id_00002/
        │   ├── img2.jpg
        │   └── ...
        └── ...
```

其中 `id_XXXXX` 为类别名称，每个类别文件夹下存放对应的图像。

---

## 三、动漫人脸识别

当以上模型训练完毕并且特征提取完毕后，即可使用anime_face_recognizer.py识别动漫人物

本项目提供一个已经训练好的yolo模型用于提取动漫人脸

请在anime_face_recognizer.py找到以下代码并修改这里的目录，改成你的测试目录并把需要识别的图片放到该目录下

```python
input_path = filedialog.askopenfilename(
        initialdir=r"C:\\Users\\CNLonely\\Desktop\\test11", #改成你的测试目录
        title="请选择图片文件",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
```
