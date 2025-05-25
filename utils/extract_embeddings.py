import os
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import InterpolationMode

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    state_dict = torch.load(Path("models/anime_character_recognition_best.pth"), map_location=device)
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 多尺度设置
    scales = [224, 256, 288]

    data_dir = Path(r"datasets/face_crop/train/")

    def get_transform(input_size):
        return transforms.Compose([
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    batch_size = 64
    num_workers = 12

    # 为每个尺度建立一个 DataLoader（同样的数据，应用不同 transform）
    datasets = {}
    dataloaders = {}
    for scale in scales:
        transform = get_transform(scale)
        datasets[scale] = ImageFolder(root=data_dir, transform=transform)
        dataloaders[scale] = DataLoader(datasets[scale], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_embeddings = []
    all_labels = []
    image_paths = [sample[0] for sample in datasets[scales[0]].samples]

    for batches in tqdm(zip(*[dataloaders[s] for s in scales]), total=len(dataloaders[scales[0]]), desc="多尺度特征提取"):
        multi_scale_feats = []

        for imgs, _ in batches:
            imgs = imgs.to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    feats = model(imgs).cpu().numpy()
            multi_scale_feats.append(feats)

        fused_feats = np.mean(multi_scale_feats, axis=0)
        norm_feats = fused_feats / np.linalg.norm(fused_feats, axis=1, keepdims=True)
        all_embeddings.append(norm_feats)

        _, labels = batches[0]
        all_labels.extend(labels.numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    os.makedirs("meta", exist_ok=True)
    np.save("meta/embeddings.npy", all_embeddings)
    np.save("meta/labels.npy", all_labels)
    with open("meta/image_paths.txt", "w", encoding="utf-8") as f:
        for p in image_paths:
            f.write(p + "\n")

    print("✅ 多尺度融合特征提取完成，已保存至 meta/")

if __name__ == "__main__":
    main()
