import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
warnings.filterwarnings('ignore')

# Import model and dataset definitions from separate files
from models.model_definitions import NIHResNet18, NIHResNet50, NIHViTModel, NIHSwinViTModel, CheXpertResNet18
from dataset.dataset_definitions import NIHDataset, CheXpertDataset

class SimpleGradCAM:
    """Simplified Grad-CAM implementation"""
    
    def __init__(self, model, target_layer, device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM for the input image"""
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(model_output)
        # 兼容 int 和 Tensor 类型
        if hasattr(target_class, 'item'):
            class_idx = target_class.item()
        else:
            class_idx = int(target_class)
        one_hot[0, class_idx] = 1
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate weights
        if self.gradients is not None and self.activations is not None:
            weights = torch.mean(self.gradients, dim=[2, 3])
            cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * self.activations, dim=1)
            cam = F.relu(cam)
            
            # Normalize
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.detach().cpu().numpy()
        else:
            return np.zeros((1, 224, 224))

class ModelLoader:
    """Load different model architectures"""
    
    def __init__(self, model_type, checkpoint_path, num_classes=14, device='cuda'):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.device = device
        self.model = None
        self.target_layer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model based on architecture type"""
        if self.model_type.lower() == 'resnet18':
            self.model = NIHResNet18(num_classes=self.num_classes)
            self.target_layer = self.model.model.layer4[-1]
            
        elif self.model_type.lower() == 'resnet50':
            self.model = NIHResNet50(num_classes=self.num_classes)
            self.target_layer = self.model.model.layer4[-1]
            
        elif self.model_type.lower() == 'vit':
            self.model = NIHViTModel(num_classes=self.num_classes)
            # For ViT, use the last attention block
            self.target_layer = self.model.img_model.encoder.layers[-1]
            
        elif self.model_type.lower() == 'swin-vit':
            self.model = NIHSwinViTModel(num_classes=self.num_classes)
            # For Swin-ViT, use the last stage
            self.target_layer = self.model.img_model.stages[-1]
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Load checkpoint
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel state dict keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            self.model.load_state_dict(new_state_dict)
            print(f"Loaded checkpoint from: {self.checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def get_target_layer(self):
        """Get the target layer for Grad-CAM"""
        return self.target_layer

class GradCAMVisualizer:
    """Main class for Grad-CAM visualization"""
    
    def __init__(self, model_type, checkpoint_path, dataset_type='nih', 
                 data_dir=None, num_classes=14, device='cuda'):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.device = device
        
        # Load model
        self.model_loader = ModelLoader(model_type, checkpoint_path, num_classes, device)
        self.model = self.model_loader.model
        self.target_layer = self.model_loader.get_target_layer()
        
        # Initialize Grad-CAM
        self.gradcam = SimpleGradCAM(self.model, self.target_layer, device)
        
        # Setup transforms
        self._setup_transforms()
        
        # Setup dataset
        self._setup_dataset()
        
        # Class names
        self.class_names = self._get_class_names()

    def _setup_transforms(self):
        """Setup image transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _setup_dataset(self):
        """Setup dataset based on type"""
        if self.dataset_type.lower() == 'chexpert':
            self.dataset = CheXpertDataset(
                root_dir=self.data_dir or "/home/mxz3935/dataset_folder/chexpert_v1.0_small",
                dataset_type="test",
                policy="zeros",
                transform=self.transform
            )
        elif self.dataset_type.lower() == 'nih':
            self.dataset = NIHDataset(
                root_dir=self.data_dir or "/home/mxz3935/dataset_folder/chest_x-ray_nih",
                dataset_type="test",
                transform=self.transform
            )
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
    def _get_class_names(self):
        """Get class names based on dataset"""
        if self.dataset_type.lower() == 'chexpert':
            return ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                   'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        elif self.dataset_type.lower() == 'nih':
            return ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate", "Mass",
                   "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
                   "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
        else:
            return [f"Class_{i}" for i in range(self.num_classes)]
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor, image
    
    def generate_heatmap(self, image_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        cam = self.gradcam.generate_cam(image_tensor, target_class)
        cam_resized = cv2.resize(cam[0], (224, 224))
        return cam_resized
    
    def overlay_heatmap(self, image, heatmap, alpha=0.3):
        """Overlay heatmap on original image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image = image.astype(np.float32) / 255.0
        
        # 使用matplotlib的plasma颜色映射，确保与colorbar一致
        # 获取plasma颜色映射
        plasma_cmap = cm.get_cmap('plasma')
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_colored = plasma_cmap(heatmap_normalized)[:, :, :3]  # 只取RGB通道
        
        overlay = alpha * heatmap_colored + (1 - alpha) * image
        return overlay, heatmap_normalized  # 返回归一化的热力图用于colorbar
    
    def visualize_single_image(self, image_path, target_class=None, save_path=None):
        """Visualize Grad-CAM for a single image"""
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Generate heatmap
        heatmap = self.generate_heatmap(image_tensor, target_class)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.sigmoid(output)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Overlay heatmap
        overlay, heatmap_normalized = self.overlay_heatmap(original_image, heatmap)
        
        # Create visualization with colorbar on overlay
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image\nFile: {os.path.basename(image_path)}')
        axes[0].axis('off')
        
        # Overlay with colorbar
        im = axes[1].imshow(overlay)
        axes[1].set_title(f'Grad-CAM Overlay\nPredicted: {self.class_names[predicted_class]}\nConfidence: {confidence:.3f}')
        axes[1].axis('off')

        # Create a mappable object for the colorbar and an axis for it
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Attention Score', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'heatmap': heatmap,
            'overlay': overlay
        }
    
    def batch_visualization(self, num_samples=5, save_dir='gradcam_batch'):
        """Generate Grad-CAM for a batch of samples"""
        os.makedirs(save_dir, exist_ok=True)
        
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        
        for i, (images, labels, image_names) in enumerate(tqdm(dataloader, desc="Generating Grad-CAM")):
            if i >= num_samples:
                break
            
            image_name = image_names[0].replace('.png', '')  # 移除.png后缀
            
            images = images.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                output = self.model(images)
                probabilities = torch.sigmoid(output)
                predicted_class = probabilities.argmax(dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Generate heatmap
            heatmap = self.generate_heatmap(images, target_class=predicted_class)
            
            # Convert tensor to PIL image for overlay
            image_tensor = images[0].cpu()
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = image_tensor * std + mean
            image_tensor = torch.clamp(image_tensor, 0, 1)
            image = transforms.ToPILImage()(image_tensor)
            
            # Overlay heatmap
            overlay, heatmap_normalized = self.overlay_heatmap(image, heatmap)
            
            # Get original labels
            original_labels = labels[0].cpu().numpy()
            label_names = []
            for j, label_val in enumerate(original_labels):
                if label_val > 0.5:  # 假设阈值是0.5
                    label_names.append(self.class_names[j])
            
            # Create visualization with colorbar on overlay
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title(f'Original Image\nLabels: {", ".join(label_names) if label_names else "No Finding"}\nFile: {image_names[0]}')
            axes[0].axis('off')
            
            # Overlay with colorbar
            im = axes[1].imshow(overlay)
            axes[1].set_title(f'Grad-CAM Overlay\nPredicted: {self.class_names[predicted_class]}\nConfidence: {confidence:.3f}')
            axes[1].axis('off')

            # Create a mappable object for the colorbar and an axis for it
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
            sm.set_array([])
            
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Attention Score', rotation=270, labelpad=15)
            
            plt.tight_layout()
            
            # 命名格式: {dataset}_{image_name}_{predicted_disease}.png
            predicted_disease = self.class_names[predicted_class]
            save_filename = f"{self.dataset_type}_{image_name}_{predicted_disease}.png"
            save_path = os.path.join(save_dir, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization for Chest X-ray Models')
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['resnet18', 'resnet50', 'vit', 'swin-vit'],
                       help='Model architecture type')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to model checkpoint (if not specified, will use default path)')
    parser.add_argument('--dataset_type', type=str, default='nih',
                       choices=['chexpert', 'nih'],
                       help='Dataset type')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset directory')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to a single image for visualization')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize from the test set')
    parser.add_argument('--save_dir', type=str, default='gradcam_results',
                       help='Directory to save visualization results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set default checkpoint path if not specified
    if args.checkpoint_path is None:
        if args.dataset_type.lower() == 'chexpert':
            args.checkpoint_path = f"checkpoints/chexpert_{args.model_type}_best.pth"
        else:
            args.checkpoint_path = f"checkpoints/nih_{args.model_type}_best.pth"
    
    # Initialize visualizer
    visualizer = GradCAMVisualizer(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        device=args.device
    )
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    if args.image_path:
        # Visualize a single image
        image_name = os.path.splitext(os.path.basename(args.image_path))[0]
        save_path = os.path.join(args.save_dir, f"{args.dataset_type}_{image_name}_gradcam.png")
        visualizer.visualize_single_image(args.image_path, save_path=save_path)
    else:
        # Visualize a batch of images
        visualizer.batch_visualization(num_samples=args.num_samples, save_dir=args.save_dir)

if __name__ == '__main__':
    main() 