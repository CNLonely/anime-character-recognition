import os
import json
import re


base_dir = r"C:\Users\CNLonely\Desktop\imgToAnime\anime\被勇者队伍开除的驯兽师，邂逅最强种猫耳少女\被勇者队伍开除的驯兽师，邂逅最强种猫耳少女"

current_dir = os.path.dirname(os.path.abspath(__file__))
meta_json_path = os.path.join(current_dir, "meta", "character_meta.json")


anime_name = os.path.basename(base_dir)

if os.path.exists(meta_json_path):
    with open(meta_json_path, "r", encoding="utf-8") as f:
        characters_info = json.load(f)
else:
    characters_info = {}

max_id_num = 0
pattern = re.compile(r"id_(\d{5})")
for key in characters_info.keys():
    match = pattern.match(key)
    if match:
        num = int(match.group(1))
        if num > max_id_num:
            max_id_num = num

subdirectories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for i, subdir in enumerate(subdirectories, start=max_id_num + 1):
    new_id = f"id_{i:05d}"

    if new_id in characters_info:
        continue

    characters_info[new_id] = {
        "name": subdir,
        "anime": anime_name
    }

    old_dir_path = os.path.join(base_dir, subdir)
    new_dir_path = os.path.join(base_dir, new_id)

    os.rename(old_dir_path, new_dir_path)

with open(meta_json_path, "w", encoding="utf-8") as f:
    json.dump(characters_info, f, ensure_ascii=False, indent=4)

print(f"角色重命名完成，信息已更新至：{meta_json_path}")
