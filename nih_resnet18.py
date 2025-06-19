import os
import time
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torchvision.models import resnet18, ResNet18_Weights

# 推荐使用 DEFAULT，始终使用最新的默认权重
model = resnet18(weights=ResNet18_Weights.DEFAULT)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ========== Environment Setup ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/nih_resnet18.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s: %(message)s",
                    filemode="w")  # Cover model

# ========== Dataset ==========
data_dir = "/home/mxz3935/dataset_folder/chest_x-ray_nih/"

class NIHDataset(Dataset):
    def __init__(self, root_dir, dataset_type="train", num_classes=14, transform=None):
        if dataset_type not in ["train", "val", "test"]:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'.")

        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.transform = transform

        self.image_dir = os.path.join(root_dir, "images")
        self.index_dir = os.path.join("./files/nih14_label", f"{dataset_type}_label.csv")

        self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, 1:self.num_classes + 1].values
        self.label_index = pd.read_csv(self.index_dir, header=0).iloc[:, :self.num_classes + 1]

    def __len__(self):
        return len(self.label_index)

    def __getitem__(self, idx):
        name = self.label_index.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_index.iloc[idx, 1:self.num_classes + 1].values.astype("int")
        return image, torch.tensor(label, dtype=torch.float32), name

# ========== Transforms ==========
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== Dataloaders ==========
train_loader = DataLoader(NIHDataset(data_dir, "train", transform=train_transforms), batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(NIHDataset(data_dir, "val", transform=val_transforms), batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(NIHDataset(data_dir, "test", transform=val_transforms), batch_size=64, shuffle=False, num_workers=4)

# ========== Model ==========
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = ResNet18(num_classes=14)
model = nn.DataParallel(model).to(devices)
logging.info(model)

# ========== Optimizer & Loss ==========
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
criterion = nn.BCEWithLogitsLoss()

# ========== Evaluation Function ==========
def evaluate(model, loader, criterion, phase="val"):
    model.eval()
    output_list, label_list = [], []

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(devices), labels.to(devices)
            logits = model(images)
            outputs = torch.sigmoid(logits)

            output_list.append(outputs.cpu().numpy())
            label_list.append(labels.cpu().numpy())

    output_array = np.concatenate(output_list)
    label_array = np.concatenate(label_list)

    aucs, accs, f1s = [], [], []
    for k in range(len(classes)):
        try:
            auc = roc_auc_score(label_array[:, k], output_array[:, k])
        except:
            auc = float("nan")
        pred = (output_array[:, k] >= 0.5).astype(int)
        acc = accuracy_score(label_array[:, k], pred)
        f1 = f1_score(label_array[:, k], pred, zero_division=0)

        aucs.append(auc)
        accs.append(acc)
        f1s.append(f1)

        logging.info(f"{phase.upper()}|{classes[k]}: AUC={auc:.4f}, ACC={acc:.4f}, F1={f1:.4f}")

    avg_auc = np.nanmean(aucs)
    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1s)
    print(f"{phase.upper()}|Avg AUC={avg_auc:.4f}, ACC={avg_acc:.4f}, F1={avg_f1:.4f}")
    logging.info(f"{phase.upper()}|Avg AUC={avg_auc:.4f}, ACC={avg_acc:.4f}, F1={avg_f1:.4f}")
    return avg_auc

# ========== Training ==========
best_auc = 0.0
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate", "Mass",
           "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
           "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

start_time = time.time()
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(devices), labels.to(devices)
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            logging.info(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print(f"[Epoch {epoch+1}] Training Loss: {running_loss / len(train_loader):.4f}")
    val_auc = evaluate(model, val_loader, criterion, phase="val")
    print(f"[Epoch {epoch+1}] Validation AUC: {val_auc:.4f}\n")
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "checkpoints/nih_resnet18_best.pth")
        print(f"Saved best model at epoch {epoch+1} with AUC: {best_auc:.4f}")
        logging.info(f"Saved best model at epoch {epoch+1} with AUC {best_auc:.4f}")

print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
print(f"Total Training Time: {(time.time() - start_time)/60:.2f} minutes")
logging.info(f"Total Training Time: {(time.time() - start_time)/60:.2f} min")

# ========== Test ==========
model.load_state_dict(torch.load("checkpoints/nih_resnet18_best.pth", weights_only = True))
evaluate(model, test_loader, criterion, phase="test")