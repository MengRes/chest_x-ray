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
from sklearn.metrics import roc_auc_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(devices)
# logging
logging.basicConfig(filename='log/simple.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

data_dir = "/home/mxz3935/dataset_folder/chest_x-ray_nih/"

class NIHDataset(Dataset):

    def __init__(self, root_dir, dataset_type = "train", num_classes = 8, img_size = 512, transform = None):
        if dataset_type not in ["train", "val", "test"]:
            raise ValueError("No such type, must be 'train', 'val', or 'test'")
        self.img_size = img_size
        self.num_classes = num_classes
        self.disease_categories = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate", "Mass",
                                   "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
                                   "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
        self.dataset_type = dataset_type
        self.image_dir = os.path.join(root_dir, "images")
        self.transform = transform
        if self.dataset_type in ["train", "val", "test"]:
            self.index_dir = os.path.join("./database/nih14_label", dataset_type + "_label.csv")
            # obtain findings classes
            self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, 1:self.num_classes + 1].values
            # obtain image name and label
            self.label_index = pd.read_csv(self.index_dir, header=0).iloc[:, :self.num_classes + 1]


    def __len__(self):
        return int(len(self.label_index))

    def __getitem__(self, idx, ):
        # obtain images name
        name = self.label_index.iloc[idx, 0]
        # obtain image location
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert("L")
        image = np.stack((image,) * 3, axis=-1)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        # obtain image label，shape: 1*num_classes 
        if self.dataset_type in ['train', 'val', 'test']:
            label = self.label_index.iloc[idx, 1:self.num_classes + 1].values.astype('int')
        return image, label, name
    
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_loader = DataLoader(NIHDataset(data_dir, dataset_type = "train", num_classes = 14, 
                                     img_size = 256, transform = train_transforms), 
                                     batch_size=64, num_workers=4, shuffle=True)
val_loader = DataLoader(NIHDataset(data_dir, dataset_type = "val", num_classes = 14, 
                                   img_size = 256, transform = val_transforms), 
                                   batch_size=64, num_workers=4, shuffle=False)


class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        img_model = models.resnet18(pretrained = True)
        img_model.fc = nn.Linear(512, num_classes)  
        self.img_model = img_model
    
    def forward(self, x):
        x = self.img_model(x)
        return(x)
    
model = ResNet18(num_classes=14)
model = nn.DataParallel(model).to(devices)
print(model)
logging.info(model)

num_epochs = 10
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate", "Mass",
           "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
           "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss() 

best_auc = 0.0  # 记录最佳 AUC
best_epoch = 0  # 记录最佳 AUC 所在 epoch
start_time = time.time()
for epoch in range(num_epochs):
    
    model.train()
    for batch_idx, data in enumerate(train_loader):
        images, labels, names = data
        images = images.to(devices)
        labels = labels.to(devices)
            
        ### FORWARD AND BACK PROP
        logits = model(images)
        outputs = logits
        # outputs = torch.sigmoid(logits)
        loss = criterion(outputs, torch.as_tensor(labels, dtype=torch.float))
        
        optimizer.zero_grad()
        loss.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), loss))
            logging.info('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                         %(epoch+1, num_epochs, batch_idx, 
                           len(train_loader), loss))

    model.eval()
    output_list = []
    label_list = []
    epoch_auc_list = []
    
    for batch_idx, data in enumerate(val_loader):
        images, labels, names = data
        images = images.cuda()
        labels = labels.cuda()
            
        ### FORWARD AND BACK PROP
        logits = model(images)
        outputs = logits
        # outputs = torch.sigmoid(logits)
        loss = criterion(outputs, torch.as_tensor(labels, dtype=torch.float))

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        for i in range(outputs.shape[0]):
            output_list.append(outputs[i].tolist())
            label_list.append(labels[i].tolist())
    
    output_array = np.array(output_list)
    label_array = np.array(label_list)

    epoch_auc_list = []
    epoch_acc_list = []

    # the average loss of one epoch
    # the average AUC of on epoch   
    for k in range(14):
        # 计算 AUC
        try:
            epoch_auc = roc_auc_score(label_array[:, k], output_array[:, k])
        except ValueError:
            epoch_auc = float("nan")  # 如果 AUC 计算失败（比如单类数据），返回 NaN
        epoch_auc_list.append(epoch_auc)

        # 计算 ACC (使用 0.5 作为阈值)
        preds = (output_array[:, k] >= 0.5).astype(int)
        epoch_acc = accuracy_score(label_array[:, k], preds)
        epoch_acc_list.append(epoch_acc)

        print("Validation|AUC of {}: {:.4f}, ACC: {:.4f}".format(classes[k], epoch_auc, epoch_acc))
        logging.info("Validation|AUC of {}: {:.4f}, ACC: {:.4f}".format(classes[k], epoch_auc, epoch_acc))

    # 计算 AUC 和 ACC 的平均值（忽略 NaN 值）
    epoch_auc_avg = np.nanmean(epoch_auc_list)
    epoch_acc_avg = np.nanmean(epoch_acc_list)

    print("Validation|Average AUC: {:.4f}, Average ACC: {:.4f}".format(epoch_auc_avg, epoch_acc_avg))
    logging.info("Validation|Average AUC: {:.4f}, Average ACC: {:.4f}".format(epoch_auc_avg, epoch_acc_avg))

    # **保存最佳 AUC 的 checkpoint**
    if epoch_auc_avg > best_auc:
        best_auc = epoch_auc_avg
        best_epoch = epoch + 1  # epoch 从 1 开始
        torch.save(model.state_dict(), "checkpoints/best.pth")
        print(f"Best model saved at epoch {best_epoch} with AUC: {best_auc:.4f}")
        logging.info(f"Best model saved at epoch {best_epoch} with AUC: {best_auc:.4f}")

    # 输出最佳 AUC 及其对应的 epoch
    print(f"Highest Validation AUC: {best_auc:.4f} at epoch {best_epoch}")
    logging.info(f"Highest Validation AUC: {best_auc:.4f} at epoch {best_epoch}")

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
logging.info('Total Training Time: %.2f min' % ((time.time() - start_time)/60))


test_loader = DataLoader(NIHDataset(data_dir, dataset_type = "test", num_classes = 14, 
                            img_size = 256, transform = val_transforms), batch_size=16, num_workers=4, shuffle=False)
model.eval()
output_list = []
label_list = []
epoch_auc_list = []
epoch_acc_list = []


for batch_idx, data in enumerate(test_loader):
    images, labels, names = data
    images = images.to(devices)
    labels = labels.to(devices)
        
    ### FORWARD
    logits = model(images)
    outputs = torch.sigmoid(logits)
    loss = criterion(outputs, torch.as_tensor(labels, dtype=torch.float))

    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    for i in range(outputs.shape[0]):
        output_list.append(outputs[i].tolist())
        label_list.append(labels[i].tolist())

output_array = np.array(output_list)
label_array = np.array(label_list)

# 计算每个类别的 AUC 和 ACC
for k in range(14):
    # 计算 AUC
    try:
        epoch_auc = roc_auc_score(label_array[:, k], output_array[:, k])
    except ValueError:
        epoch_auc = float("nan")  # AUC 计算失败时返回 NaN
    epoch_auc_list.append(epoch_auc)

    # 计算 ACC (使用 0.5 作为阈值)
    preds = (output_array[:, k] >= 0.5).astype(int)
    epoch_acc = accuracy_score(label_array[:, k], preds)
    epoch_acc_list.append(epoch_acc)

    print("Test|AUC of {}: {:.4f}, ACC: {:.4f}".format(classes[k], epoch_auc, epoch_acc))
    logging.info("Test|AUC of {}: {:.4f}, ACC: {:.4f}".format(classes[k], epoch_auc, epoch_acc))

# 计算 AUC 和 ACC 的平均值（忽略 NaN）
epoch_auc_avg = np.nanmean(epoch_auc_list)
epoch_acc_avg = np.nanmean(epoch_acc_list)

print("Test|Average AUC: {:.4f}, Average ACC: {:.4f}".format(epoch_auc_avg, epoch_acc_avg))
logging.info("Test|Average AUC: {:.4f}, Average ACC: {:.4f}".format(epoch_auc_avg, epoch_acc_avg))