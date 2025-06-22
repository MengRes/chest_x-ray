import torch
import torch.nn as nn
import torchvision.models as models

class NIHResNet18(nn.Module):
    """ResNet18 model for NIH dataset"""
    def __init__(self, num_classes=14):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class NIHResNet50(nn.Module):
    """ResNet50 model for NIH dataset"""
    def __init__(self, num_classes=14):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class NIHViTModel(nn.Module):
    """ViT model for NIH dataset"""
    def __init__(self, num_classes=14):
        super().__init__()
        img_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        img_model.heads = nn.Linear(img_model.heads.head.in_features, num_classes)
        self.img_model = img_model
    
    def forward(self, x):
        return self.img_model(x)

class NIHSwinViTModel(nn.Module):
    """Swin-ViT model for NIH dataset"""
    def __init__(self, num_classes=14):
        super().__init__()
        img_model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
        img_model.head = nn.Linear(img_model.head.in_features, num_classes)
        self.img_model = img_model
    
    def forward(self, x):
        return self.img_model(x)

class CheXpertResNet18(nn.Module):
    """ResNet18 model for CheXpert dataset"""
    def __init__(self, num_classes=14):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x) 