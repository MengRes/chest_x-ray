# Chest X-ray Classification and Grad-CAM Visualization

This project provides a comprehensive toolkit for training and evaluating deep learning models for chest X-ray image classification. It includes support for multiple popular model architectures and datasets, along with a powerful Grad-CAM visualization tool to interpret model predictions.

## Features

- **Multiple Model Architectures**: Supports `ResNet-18`, `ResNet-50`, `Vision Transformer (ViT)`, and `Swin Transformer (Swin-ViT)`.
- **Multiple Datasets**: Integrated support for the `NIH Chest X-ray 14` and `CheXpert` datasets.
- **Decoupled Definitions**: Model and dataset class definitions are separated into `models/model_definitions.py` and `dataset/dataset_definitions.py` for clarity and modularity.
- **Interpretability**: Generates Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations to highlight the regions in an image that are most important for a model's prediction.
- **Configurable Training**: Each model architecture has its own training script (e.g., `nih_resnet18.py`).
- **Command-Line Interface**: A flexible CLI for generating visualizations for single images or batches of images from the test set.

## Project Structure

```
chest_x-ray/
│
├── checkpoints/              # Stores trained model weights (e.g., nih_resnet18_best.pth)
├── dataset/
│   ├── __init__.py
│   └── dataset_definitions.py  # Contains NIHDataset and CheXpertDataset classes
│
├── files/                    # Contains label files for datasets
├── grad-cam_visualization.py # Main script for generating Grad-CAM visualizations
├── gradcam_results/          # Default output directory for saved visualizations
│
├── models/
│   ├── __init__.py
│   └── model_definitions.py    # Contains all model architecture classes
│
├── nih_resnet18.py           # Example training script for ResNet-18 on NIH-14
├── chexpert_resnet18.py      # Example training script for ResNet-18 on CheXpert
│
└── README.md
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd chest_x-ray
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Datasets:**
    Download the `NIH Chest X-ray 14` and/or `CheXpert` datasets and place them in your preferred directory. You will need to specify the root directory of the dataset when running the visualization script if it's not in the default location assumed by the script.

4.  **Place Checkpoints:**
    Ensure your pre-trained model checkpoints are placed in the `checkpoints/` directory. The scripts use a default naming convention (e.g., `nih_resnet18_best.pth`, `chexpert_vit_best.pth`).

## Usage: Grad-CAM Visualization

The `grad-cam_visualization.py` script is used to generate and save Grad-CAM heatmaps.

### Basic Command

The script requires specifying the model architecture and the dataset type.

```bash
python grad-cam_visualization.py --model_type <model> --dataset_type <dataset>
```

### Arguments

- `--model_type`: (Required) The model architecture. Choices: `resnet18`, `resnet50`, `vit`, `swin-vit`.
- `--dataset_type`: (Required) The dataset the model was trained on. Choices: `nih`, `chexpert`.
- `--checkpoint_path`: (Optional) Path to a specific model checkpoint. If not provided, a default path based on model and dataset type will be used (e.g., `checkpoints/nih_resnet18_best.pth`).
- `--data_dir`: (Optional) Path to the root directory of the dataset.
- `--image_path`: (Optional) Path to a single image file to visualize. If provided, the script will run on this image only.
- `--num_samples`: (Optional) Number of random samples to visualize from the test set. Default is `5`.
- `--save_dir`: (Optional) Directory to save the output images. Default is `gradcam_results/`.
- `--device`: (Optional) Device to run the model on. Choices: `cuda`, `cpu`. Default is `cuda`.

### Examples

**1. Visualize 5 random samples from the NIH test set using a ResNet-18 model:**
```bash
python grad-cam_visualization.py --model_type resnet18 --dataset_type nih --num_samples 5 --device cpu
```

**2. Visualize a single specific image:**
```bash
python grad-cam_visualization.py --model_type resnet18 --dataset_type nih --image_path /path/to/your/image.png --device cpu
```

**3. Visualize using a ViT model trained on CheXpert:**
```bash
python grad-cam_visualization.py --model_type vit --dataset_type chexpert --device cpu
```

## Output

The script will generate side-by-side comparisons of the original image and the image with the Grad-CAM overlay. The results are saved in the directory specified by `--save_dir` (default: `gradcam_results/`).

The output filename is structured as: `{dataset_type}_{original_image_name}_{predicted_disease}.png`.

