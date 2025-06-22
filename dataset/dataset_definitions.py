import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class NIHDataset(Dataset):
    """NIH Chest X-ray dataset"""
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

class CheXpertDataset(Dataset):
    """CheXpert dataset"""
    def __init__(self, root_dir, dataset_type="train", policy="zeros", transform=None):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.policy = policy
        self.transform = transform
        
        # Load labels
        label_file = os.path.join("./files/chexpert_label", f"split_{dataset_type}.csv")
        self.labels_df = pd.read_csv(label_file)
        
        # Filter out missing images
        self.labels_df = self.labels_df[self.labels_df['Path'].notna()]
        
        # Define label columns (excluding Path and other metadata)
        self.label_columns = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
                             'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
                             'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
                             'Pleural Other', 'Fracture', 'Support Devices']

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        # Get image path
        img_path = os.path.join(self.root_dir, row['Path'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = []
        for col in self.label_columns:
            label_val = row[col]
            if pd.isna(label_val) or label_val == -1:
                # Handle uncertain labels based on policy
                if self.policy == "zeros":
                    labels.append(0.0)
                elif self.policy == "ones":
                    labels.append(1.0)
                else:  # policy == "ignore"
                    labels.append(0.0)
            else:
                labels.append(float(label_val))
        
        return image, torch.tensor(labels, dtype=torch.float32), row['Path'] 