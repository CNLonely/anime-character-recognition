import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import json
import os
from pathlib import Path
from collections import defaultdict
import random
from torchvision.transforms import InterpolationMode
from PIL import Image
from timm.data.auto_augment import rand_augment_transform
from sklearn.metrics import recall_score  
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F  
import pandas as pd
import numpy as np 

correct_counts = {}
total_counts = {}

val_correct_counts = {}
val_total_counts = {}


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):#0.1
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class ClassAwareSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_samples_cls=1):
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.imgs):
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())
        self.num_samples_cls = num_samples_cls
        self.num_samples = len(dataset)

    def __iter__(self):
        sampled_indices = []
        class_iters = {
            cls: iter(random.choices(indices, k=self.num_samples_cls * len(self.classes)))
            for cls, indices in self.class_to_indices.items()
        }
        while len(sampled_indices) < self.num_samples:
            cls = random.choice(self.classes)
            try:
                for _ in range(self.num_samples_cls):
                    sampled_indices.append(next(class_iters[cls]))
            except StopIteration:
                continue
        return iter(sampled_indices[:self.num_samples])

    def __len__(self):
        return self.num_samples

class ComicStyle(object):
  def __init__(self, threshold=100):
    self.threshold = threshold

  def __call__(self, img: Image.Image):
    gray = img.convert('L') # 灰度
    bw = gray.point(lambda x: 255 if x > self.threshold else 0, '1') # 二值化
    return bw.convert('RGB') # 转回RGB

  def __repr__(self):
    return f'{self.__class__.__name__}(threshold={self.threshold})'


class AddChannelNoise(torch.nn.Module):
  def __init__(self, std=0.02, p=0.1):
    super().__init__()
    self.std = std
    self.p = p

  def forward(self, img):
    if torch.rand(1) < self.p:
      noise = torch.randn_like(img) * self.std
      img = img + noise
      img = torch.clamp(img, 0., 1.)
    return img


def rand_bbox(size, lam):
  W = size[2]
  H = size[3]
  cut_rat = math.sqrt(1. - lam)
  cut_w = int(W * cut_rat)
  cut_h = int(H * cut_rat)

  cx = torch.randint(W, (1,)).item()
  cy = torch.randint(H, (1,)).item()

  bbx1 = max(0, min(cx - cut_w // 2, W))
  bby1 = max(0, min(cy - cut_h // 2, H))
  bbx2 = max(0, min(cx + cut_w // 2, W))
  bby2 = max(0, min(cy + cut_h // 2, H))

  return bbx1, bby1, bbx2, bby2

def cutmix(data, targets, alpha=1.0):
  lam = np.random.beta(alpha, alpha)
  rand_index = torch.randperm(data.size()[0]).to(data.device)
  target_a = targets
  target_b = targets[rand_index]
  bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
  data[:, :, bby1:bby2, bbx1:bbx2] = data[rand_index, :, bby1:bby2, bbx1:bbx2]
  lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
  return data, target_a, target_b, lam

def mixup(data, targets, alpha=1.0):
  lam = np.random.beta(alpha, alpha)
  rand_index = torch.randperm(data.size()[0]).to(data.device)
  mixed_data = lam * data + (1 - lam) * data[rand_index, :]
  target_a = targets
  target_b = targets[rand_index]
  return mixed_data, target_a, target_b, lam


def fgsm_attack(model, images, labels, epsilon=0.01, criterion=None):
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad_sign = images.grad.sign()

    with torch.no_grad():
        adv_images = images + epsilon * grad_sign
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
    return adv_images.detach()


if __name__ == '__main__':
  with open('meta/character_meta.json', 'r', encoding='utf-8') as f:
    character_meta = json.load(f)
  
  train_root = "datasets/face_crop/train"
  folder_names = sorted(os.listdir(train_root)) 
  id_map = {i: folder_names[i] for i in range(len(folder_names))}

  batch_size = 64
  epochs = 100
  learning_rate = 0.0003 # 调高初始学习率
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),

    transforms.RandomChoice([
      transforms.RandomRotation(10),
      transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=3),
    ]),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(
      brightness=0.2,
      contrast=0.2,
      saturation=0.05,
      hue=0.03
    ),
    rand_augment_transform('rand-m5-n1'),
    # transforms.RandomApply([
    #   ComicStyle(threshold=100),
    # ], p=0.1),
    transforms.ToTensor(),
    AddChannelNoise(std=0.02, p=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
    transforms.RandomApply([
      transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 1.0))
    ], p=0.2),
    transforms.RandomErasing(p=0.3, scale=(0.01, 0.1), ratio=(0.3, 3.3), value='random') 
  ])

  val_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
  ])

  train_dir = Path(r'datasets/face_crop/train/')
  val_dir = Path(r'datasets/face_crop/val/')

  train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
  val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

  num_classes = len(character_meta)

  class_counts = [0] * num_classes
  for _, label in train_dataset.imgs:
    class_counts[label] += 1

  total_samples = sum(class_counts)
  class_weights = [total_samples / count for count in class_counts]
  class_weights = torch.tensor(class_weights).to(device)

  criterion = LabelSmoothingLoss(smoothing=0.1)

  sampler = ClassAwareSampler(train_dataset, num_samples_cls=1)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=16, persistent_workers=True, pin_memory=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, persistent_workers=True, pin_memory=True)

  model = models.resnet50(pretrained=True)
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  model = model.to(device)

  optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  checkpoint_dir = 'models/'
  os.makedirs(checkpoint_dir, exist_ok=True)

  best_val_acc = 0.0
  patience = 5
  early_stop_counter = 0

  scaler = torch.amp.GradScaler('cuda') 

  def train_one_epoch_adv(epsilon=0.01, alpha=0.5, use_cutmix=True, use_mixup=True):
      global correct_counts, total_counts
      correct_counts.clear()
      total_counts.clear()
      model.train()
      running_loss = 0.0
      correct = 0
      total = 0

      all_preds = []
      all_targets = []

      top5_correct = 0 

      for images, labels in tqdm(train_loader, desc="Training"):
          images, labels = images.to(device), labels.to(device)

          if use_cutmix and random.random() < 0.5:
              images, targets_a, targets_b, lam = cutmix(images, labels)
          elif use_mixup and random.random() < 0.5:
              images, targets_a, targets_b, lam = mixup(images, labels)
          else:
              targets_a, targets_b, lam = labels, labels, 1.0

          optimizer.zero_grad()
          with torch.amp.autocast('cuda'):
              outputs = model(images)
              if lam == 1.0:
                  loss = criterion(outputs, labels)
              else:
                  loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

              adv_images = fgsm_attack(model, images, targets_a, epsilon, criterion)
              adv_outputs = model(adv_images)
              adv_loss = lam * criterion(adv_outputs, targets_a) + (1 - lam) * criterion(adv_outputs, targets_b)

              total_loss = alpha * loss + (1 - alpha) * adv_loss

          scaler.scale(total_loss).backward()
          scaler.step(optimizer)
          scaler.update()

          running_loss += total_loss.item()

          _, predicted = torch.max(outputs, 1)
          correct += (predicted == labels).sum().item()
          total += labels.size(0)

          _, top5_pred = outputs.topk(5, dim=1)
          top5_correct += (top5_pred == labels.unsqueeze(1)).sum().item()

          preds_np = predicted.cpu().numpy()
          labels_np = labels.cpu().numpy()
          for p, gt in zip(preds_np, labels_np):
              total_counts[gt] = total_counts.get(gt, 0) + 1
              if p == gt:
                  correct_counts[gt] = correct_counts.get(gt, 0) + 1
  
          all_preds.extend(predicted.cpu().tolist())
          all_targets.extend(labels.cpu().tolist())

      epoch_loss = running_loss / len(train_loader)
      epoch_acc = 100 * correct / total
      epoch_top5_acc = 100 * top5_correct / total 

      recall = recall_score(all_targets, all_preds, average='macro') * 100

      train_class_acc = {
          i: (correct_counts.get(i, 0) / total_counts[i]) if total_counts.get(i, 0) > 0 else 0.0
          for i in total_counts.keys()
      }

      return epoch_loss, epoch_acc, epoch_top5_acc, recall, train_class_acc


  def validate():
      global val_correct_counts, val_total_counts
      val_total_counts.clear()
      val_correct_counts.clear()
      model.eval()
      correct = 0
      total = 0
      all_preds = []
      all_targets = []
      total_loss = 0.0

      top5_correct = 0  
  
      with torch.no_grad():
          for images, labels in tqdm(val_loader, desc="Validating"):
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)

              loss = criterion(outputs, labels)
              total_loss += loss.item() * labels.size(0)

              _, predicted = torch.max(outputs, 1)
              correct += (predicted == labels).sum().item()
              total += labels.size(0)

              _, top5_pred = outputs.topk(5, dim=1)
              top5_correct += (top5_pred == labels.unsqueeze(1)).sum().item()

              for p, gt in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                  val_total_counts[gt] = val_total_counts.get(gt, 0) + 1
                  if p == gt:
                      val_correct_counts[gt] = val_correct_counts.get(gt, 0) + 1

              all_preds.extend(predicted.cpu().tolist())
              all_targets.extend(labels.cpu().tolist())

      val_loss = total_loss / total
      val_acc = 100 * correct / total
      val_top5_acc = 100 * top5_correct / total  

      recall = recall_score(all_targets, all_preds, average='macro') * 100

      val_class_acc = {
          i: (val_correct_counts.get(i, 0) / val_total_counts[i]) if val_total_counts.get(i, 0) > 0 else 0.0
          for i in val_total_counts.keys()
      }

      return val_loss, val_acc, val_top5_acc, recall, val_class_acc


  for epoch in range(epochs):
    print(f"\nEpoch [{epoch+1}/{epochs}]")

    train_loss, train_acc, train_top5_acc, train_recall,train_class_acc = train_one_epoch_adv(epsilon=0.01, alpha=0.5)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Top-5 Acc: {train_top5_acc:.2f}%, Recall: {train_recall:.2f}%")

    val_loss,val_acc, val_top5_acc, val_recall,val_class_acc = validate()
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%, Top-5 Acc: {val_top5_acc:.2f}%, Recall: {val_recall:.2f}%")

    scheduler.step()

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'anime_character_recognition_best.pth'))
      print("模型已保存！")
      early_stop_counter = 0

      class_ids = list(id_map.keys())
      role_names = [id_map[i] for i in class_ids]
      train_acc_list = [round(train_class_acc.get(i, 0.0), 2) for i in class_ids]
      val_acc_list = [round(val_class_acc.get(i, 0.0), 2) for i in class_ids]
  
      df = pd.DataFrame({
        "角色名称": role_names,
        "训练准确率": train_acc_list,
        "验证准确率": val_acc_list
      })

      df["训练准确率"] = df["训练准确率"].astype(float)
      df["验证准确率"] = df["验证准确率"].astype(float)

      acc_save_path = os.path.join(checkpoint_dir, 'best_epoch_class_accuracies.csv')
      df.to_csv(acc_save_path, index=False, encoding='utf-8-sig', float_format='%.2f')

      print(f"该轮训练/验证每类准确率已保存至：{acc_save_path}")
    else:
      early_stop_counter += 1
      if early_stop_counter >= patience:
        print(f"验证精度未提升，提前停止训练！")
        break

  print("训练完成！")
  class_ids = list(id_map.keys())
  class_names = [id_map[i] for i in class_ids]
  accuracies = [
    (correct_counts.get(i, 0) / total_counts[i]) if total_counts.get(i, 0) > 0 else 0.0
    for i in class_ids
  ]

  low_acc_pairs = [
    (acc, name) for acc, name in zip(accuracies, class_names) if acc < 0.7
  ]

  if not low_acc_pairs:
    print("所有角色准确率都高于70%！")
  else:
    low_acc_pairs.sort(key=lambda x: x[0])
    low_accs, low_names = zip(*low_acc_pairs)

    plt.figure(figsize=(10, max(4, len(low_accs) * 0.4)))
    plt.barh(low_names, low_accs, color='orange')
    plt.xlabel("准确率")
    plt.title("准确率低于70%的角色")
    plt.xlim(0, 1)
    
    fontsize = max(6, min(12, 12 - len(low_accs) // 10))
    plt.tick_params(axis='y', labelsize=fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)

    plt.tight_layout()

    plt.close()