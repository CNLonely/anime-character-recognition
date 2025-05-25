import os
import shutil
import random
from tqdm import tqdm
from pathlib import Path

# 设置固定的随机种子
random.seed(42)

# 原始数据目录
input_dir = Path(r"datasets/face_crop/full/")
output_root = "datasets/face_crop/"


split_ratio = {
    "train": 0.8,
    "val": 0.2,
}

# 最小限制
min_total_samples = 10
min_val_samples = 1

# 清空旧数据
for split in split_ratio:
    split_path = os.path.join(output_root, split)
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
    os.makedirs(split_path, exist_ok=True)

# 获取类别
classes = [d for d in os.listdir(input_dir) if os.path.isdir(input_dir / d)]

# 数据划分
for cls in tqdm(classes, desc="划分中"):
    cls_dir = input_dir / cls
    imgs = os.listdir(cls_dir)
    random.shuffle(imgs)  # 打乱数据
    total = len(imgs)

    split_data = {
        "train": [],
        "val": [],
    }

    if total < min_total_samples:
        # 小类别：全部用于训练，复制几张用于验证
        split_data["train"] = imgs[:]
        split_data["val"] = imgs[:min(min_val_samples, total)]
    else:
        # 正常划分
        val_count = max(int(total * split_ratio["val"]), min_val_samples)
        train_count = total - val_count

        split_data["train"] = imgs[:train_count]
        split_data["val"] = imgs[train_count:train_count + val_count]

    # 保存文件
    for split, img_list in split_data.items():
        split_dir = os.path.join(output_root, split, cls)
        os.makedirs(split_dir, exist_ok=True)
        for img in img_list:
            src = os.path.join(input_dir, cls, img)
            dst = os.path.join(split_dir, img)

            # 小样本验证集复制，其它正常拷贝
            if split == "val" and total < min_total_samples:
                shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)

print("数据集划分完成！验证集比例为 20%，小样本保留训练数据")