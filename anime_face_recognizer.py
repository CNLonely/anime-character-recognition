import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from torchvision import transforms, models
import faiss
import json
from tkinter import filedialog, Tk

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载模型 ==========
# YOLOv8 人脸检测模型
yolo_model = YOLO("models/best.pt")


character_model = models.resnet50(pretrained=False)
character_model.fc = torch.nn.Identity()
state_dict = torch.load("models/anime_character_recognition_best.pth", map_location=device)
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)
character_model.load_state_dict(state_dict, strict=False)
character_model.to(device)
character_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_normalized_embedding(tensor):
    with torch.no_grad():
        emb = character_model(tensor).cpu().numpy()
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norm

# ========== FAISS加载 ==========
embeddings = np.load("meta/embeddings.npy")
labels = np.load("meta/labels.npy")
with open("meta/image_paths.txt", "r", encoding="utf-8") as f:
    image_paths = [line.strip() for line in f.readlines()]
with open("meta/character_meta.json", "r", encoding="utf-8") as f:
    character_meta = json.load(f)

folder_names = sorted(os.listdir("datasets/face_crop/train/"))
id_map = {i: name for i, name in enumerate(folder_names)}

# 构建 FAISS Index（使用余弦相似度，先归一化）
index = faiss.IndexFlatIP(embeddings.shape[1])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index.add(embeddings)

# 相似度阈值（例如 0.2 ~ 0.5，越接近1越相似）
similarity_threshold = 0.81

def draw_chinese_text(image_cv2, position, text_lines, font_path="C:/Windows/Fonts/msyh.ttc"):
    image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    image_width, image_height = image_pil.size
    base_font_size = 36
    font_size = int(base_font_size * (image_width / 1920))
    font_size = max(font_size, 20)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"字体文件 {font_path} 未找到，请检查路径")
        return image_cv2

    x, y = position
    max_width = image_cv2.shape[1]
    max_height = image_cv2.shape[0]
    line_height = font_size + 10
    text_height = len(text_lines) * line_height

    if y - text_height < 0:
        y = y + 60
    if y + text_height > max_height:
        y = max_height - text_height - 10

    text_width = max([draw.textbbox((x, y + i * line_height), line, font=font)[2] - draw.textbbox((x, y + i * line_height), line, font=font)[0] for i, line in enumerate(text_lines)])
    if x + text_width > max_width:
        x = max_width - text_width - 10

    def draw_text_with_outline(draw_obj, position, text, font, fill_color, outline_color='black', outline_width=2):
        px, py = position
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw_obj.text((px + dx, py + dy), text, font=font, fill=outline_color)
        draw_obj.text(position, text, font=font, fill=fill_color)

    for i, line in enumerate(text_lines):
        draw_text_with_outline(draw, (x, y + i * line_height), line, font, fill_color='white')

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def predict_image(image_path, topk=2, save_result=True):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model(image_rgb, conf=0.4, iou=0.5)[0]

    if len(results.boxes) == 0:
        print("未检测到人脸")
        return

    print(f"检测到 {len(results.boxes)} 张人脸\n")
    seen_labels = set()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    class_names = [results.names[cid] for cid in class_ids]

    for idx, (box, conf, cls_name) in enumerate(zip(results.boxes.xyxy.cpu().numpy(),
                                                    results.boxes.conf.cpu().numpy(),
                                                    class_names)):
        x1, y1, x2, y2 = map(int, box)
        confidence = float(conf)
        if confidence < 0.7:
            continue

        print(f"[人脸{idx+1}] 类别: {cls_name}，置信度: {confidence:.2f} 坐标: ({x1}, {y1}, {x2}, {y2})")

        face_crop = image_rgb[y1:y2, x1:x2]
        face_pil = Image.fromarray(face_crop)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        emb = get_normalized_embedding(face_tensor).reshape(1, -1)

        D, I = index.search(emb, topk * 2)

        shown = 0
        best_info = None
        best_score = 0

        for j in range(I.shape[1]):
            faiss_idx = I[0][j]
            label_id = labels[faiss_idx]
            if label_id in seen_labels:
                continue
            seen_labels.add(label_id)

            folder_id = id_map.get(label_id)
            if folder_id is None:
                continue                                                                                                            

            info = character_meta.get(folder_id, {})
            score = D[0][j]

            if score < similarity_threshold:
                print(f"  Top{shown+1}: 相似度 {score:.4f} → 该人物未知")
                continue

            if shown == 0:
                best_info = info
                best_score = score

            print(f"  Top{shown+1}: 类别{folder_id} 相似度 {score:.4f} → {info.get('name')}（{info.get('anime')}）")
            shown += 1
            if shown >= topk:
                break

        print("-" * 40)

        name = best_info.get("name", "未知") if best_info else "未知"
        anime = best_info.get("anime", "未知作品") if best_info else "未知作品"
        suffix = "（背面图）" if "back" in cls_name.lower() else ""
        label_lines = [f"{name}（{anime}）{suffix}", f"相似度: {best_score:.4f}" if best_info else "相似度: N/A"]

        if name != '未知':
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_y_position = y1 - 60 if y1 - 60 > 0 else y2 + 10
            image_bgr = draw_chinese_text(image_bgr, (x1, draw_y_position), label_lines)

    if save_result:
        save_path = os.path.splitext(image_path)[0] + "_result.jpg"
        cv2.imwrite(save_path, image_bgr)
        print(f"结果图已保存至: {save_path}")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    input_path = filedialog.askopenfilename(
        initialdir=r"C:\\Users\\CNLonely\\Desktop\\test11", #改成你的测试目录
        title="请选择图片文件",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )

    if input_path:
        if os.path.exists(input_path):
            predict_image(input_path, topk=2)
        else:
            print("输入的图片路径不存在，请检查路径")
    else:
        print("没有选择任何文件")