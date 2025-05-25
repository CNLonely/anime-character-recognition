import os
import json
import csv
from collections import defaultdict

# 路径设置
root_dir = r"C:\Users\CNLonely\Desktop\imgToAnimeInfo\datasets\face_crop\full"
meta_path = r"C:\Users\CNLonely\Desktop\imgToAnimeInfo\meta\character_meta.json"
csv_output_path = r"C:\Users\CNLonely\Desktop\filtered_characters.csv"

# 读取 character_meta.json
with open(meta_path, 'r', encoding='utf-8') as f:
    meta_data = json.load(f)

# 用于存储文件数 < 10 的角色信息
filtered_results = []

# 遍历子目录并筛选
for subdir in os.listdir(root_dir):
    full_path = os.path.join(root_dir, subdir)
    if os.path.isdir(full_path):
        file_count = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
        if file_count < 10:
            char_info = meta_data.get(subdir, {})
            anime = char_info.get("anime", "未知")
            name = char_info.get("name", "未知")
            filtered_results.append({
                "id": subdir,
                "count": file_count,
                "anime": anime,
                "name": name
            })

# 按文件数降序排序
filtered_results.sort(key=lambda x: x["count"], reverse=True)

# 保存为 CSV 文件
with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
    fieldnames = ['id', 'count', 'anime', 'name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item in filtered_results:
        writer.writerow(item)

print(f"已将筛选结果按降序保存至 CSV 文件：{csv_output_path}")
