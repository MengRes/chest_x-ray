import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import time
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/chexpert_resnet18.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s: %(message)s",
                    filemode="w")  # Cover model

data_dir = "/home/mxz3935/dataset_folder/chexpert_v1.0_small"

class CheXpertDataset(Dataset):
    """
    CheXpert Dataset for chest X-ray classification.
    
    Args:
        root_dir (str): Root directory containing the images
        dataset_type (str): Type of dataset ('train', 'val', 'test')
        num_classes (int): Number of classes to predict
        policy (str): Policy for handling uncertain labels ('zeros' or 'ones')
        transform: Optional transform to be applied to images
        filter_frontal_only (bool): Whether to filter only frontal images
    """
    def __init__(self, root_dir, dataset_type="train", num_classes=14, policy="zeros", 
                 transform=None, filter_frontal_only=True):
        if dataset_type not in ["train", "val", "test"]:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'.")
        
        if policy not in ["zeros", "ones"]:
            raise ValueError("policy must be 'zeros' or 'ones'.")

        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.policy = policy
        self.transform = transform
        self.filter_frontal_only = filter_frontal_only

        self.index_dir = os.path.join("./files/chexpert_label", f"split_{dataset_type}.csv")
        
        # Check if file exists
        if not os.path.exists(self.index_dir):
            raise FileNotFoundError(f"Label file not found: {self.index_dir}")
        
        # Read CSV file
        self.df = pd.read_csv(self.index_dir)
        
        # Filter frontal images only if requested
        if self.filter_frontal_only:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)
        
        # Get class names
        self.classes = self.df.columns[5:5+num_classes].values
        
        # Pre-process labels for better performance
        self._preprocess_labels()
        
        # Validate data
        self._validate_data()
        
        logging.info(f"Loaded {len(self.df)} samples for {dataset_type} set")

    def _preprocess_labels(self):
        """Pre-process labels to avoid repeated computation during training."""
        self.processed_labels = []
        
        for _, row in self.df.iterrows():
            labels = []
            for i in range(self.num_classes):
                label_val = row.iloc[5 + i]  # Labels start from column 6
                
                if pd.isna(label_val) or label_val == '':
                    labels.append(0)
                else:
                    label_val = float(label_val)
                    if label_val == 1:
                        labels.append(1)
                    elif label_val == -1:
                        labels.append(1 if self.policy == "ones" else 0)
                    else:
                        labels.append(0)
            
            self.processed_labels.append(labels)

    def _validate_data(self):
        """Validate the dataset for potential issues."""
        # Check for missing images and filter them out
        valid_indices = []
        missing_images = []
        
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.root_dir, str(row["Path"]))
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                missing_images.append(str(row["Path"]))
        
        # Filter dataframe to only include existing images
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        # Re-process labels for the filtered dataset
        self._preprocess_labels()
        
        if missing_images:
            logging.warning(f"Missing images: {missing_images[:5]}...")  # Log first 5 missing images
        
        # Check label distribution
        label_counts = np.sum(self.processed_labels, axis=0)
        logging.info(f"Label distribution: {dict(zip(self.classes, label_counts))}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["Path"]
        img_path = os.path.join(self.root_dir, image_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Use pre-processed labels for better performance
        labels = self.processed_labels[idx]

        return image, torch.tensor(labels, dtype=torch.float32), image_name

    def get_class_names(self):
        """Get the class names."""
        return self.classes

    def get_label_distribution(self):
        """Get the distribution of labels."""
        return np.sum(self.processed_labels, axis=0)

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

train_loader = DataLoader(CheXpertDataset(data_dir, "train", policy="zeros", transform=train_transforms, filter_frontal_only=True), 
                         batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(CheXpertDataset(data_dir, "val", policy="zeros", transform=val_transforms, filter_frontal_only=True), 
                       batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(CheXpertDataset(data_dir, "test", policy="zeros", transform=val_transforms, filter_frontal_only=True), 
                        batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final layer with dropout and new classification layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(  # type: ignore
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = ResNet18(num_classes=14)
model = nn.DataParallel(model).to(devices)
logging.info(model)

# ========== Optimizer & Loss ==========
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # type: ignore
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
        f1 = f1_score(label_array[:, k], pred, zero_division='warn')

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

best_auc = 0.0
classes = ['No Finding',
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
        'Support Devices']

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
    
    # 学习率调度
    scheduler.step(val_auc)
    
    if val_auc > best_auc:
        best_auc = val_auc
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/chexpert_resnet18_best.pth")
        print(f"Saved best model at epoch {epoch+1} with AUC: {best_auc:.4f}")
        logging.info(f"Saved best model at epoch {epoch+1} with AUC {best_auc:.4f}")

print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
print(f"Total Training Time: {(time.time() - start_time)/60:.2f} minutes")
logging.info(f"Total Training Time: {(time.time() - start_time)/60:.2f} min")

# ========== Test ==========
model.load_state_dict(torch.load("checkpoints/chexpert_resnet18_best.pth", weights_only=True))
evaluate(model, test_loader, criterion, phase="test")