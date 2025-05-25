# anime-character-recognition
训练和识别动漫人物

## 使用说明

本项目基于 ResNet-50 实现动漫角色识别，采用多种训练增强技术（CutMix、MixUp、对抗训练等）以提升模型鲁棒性和准确率。

### 一、数据准备

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

### 二、环境依赖

建议使用 Python 3.8 及以上版本，依赖的主要包如下：

在使用本项目前，请先安装依赖
```pip
pip install -r requirements.txt
```

---

### 三、训练方法

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

### 四、模型输出与日志

训练过程中将输出以下信息：

* Top-1 / Top-5 准确率
* Macro Recall 值
* 每类角色的准确率（训练集和验证集）

保存文件如下：

* 最佳模型参数：`models/anime_character_recognition_best.pth`
* 各类别准确率：`models/best_epoch_class_accuracies.csv`

训练结束后还会绘制一张图表，显示准确率低于 70% 的角色（默认不弹出界面）。

---

### 五、模型结构说明

本项目使用 ResNet-50 作为基础模型结构，加载预训练参数后将输出层替换为适配当前分类数的全连接层：

```python
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

---

### 六、增强与正则化技术

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

