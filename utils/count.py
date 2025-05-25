import json

with open('./meta/character_meta.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

anime_set = set()
for item in data.values():
    anime_set.add(item['anime'])

print(f"一共有 {len(anime_set)} 个不同的动漫。")
    