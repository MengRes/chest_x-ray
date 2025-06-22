import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# --- Plotting Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
FONT_CONFIG = {'fontsize': 12}
TITLE_CONFIG = {'fontsize': 16, 'fontweight': 'bold'}


def plot_label_distribution(df, split_name, save_dir):
    """
    Calculates and plots the distribution of disease labels for a given data split.
    """
    disease_list = df.columns[1:].tolist()
    disease_counts = df[disease_list].sum().astype(int)

    print(f"\n--- Analyzing: {split_name.upper()} Split ---")
    print(f"Total images: {len(df)}")
    print(f"Disease counts:\n{disease_counts}")

    plt.figure(figsize=(15, 7))
    bars = sns.barplot(x=disease_counts.index, y=disease_counts.values, edgecolor="k")

    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 50, int(yval),
                 ha="center", va="bottom", **FONT_CONFIG)

    plt.xticks(rotation=-30, ha='left')
    plt.xlabel("Disease Labels", **FONT_CONFIG)
    plt.ylabel("Number of Occurrences", **FONT_CONFIG)
    plt.title(f"Disease Occurrence in NIH-14 {split_name.capitalize()} Set", **TITLE_CONFIG)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"nih14_{split_name}_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved label distribution plot to: {save_path}")
    plt.close()


def analyze_sample_types(df, split_name):
    """
    Counts and prints the number of 'No Finding', 'Single-label', and 'Multi-label' images.
    """
    label_df = df.iloc[:, 1:]
    label_counts = label_df.sum(axis=1)

    no_finding = (label_counts == 0).sum()
    one_label = (label_counts == 1).sum()
    multi_label = (label_counts > 1).sum()

    print(f"Sample Types in {split_name.upper()}:")
    print(f"  - No Finding: {no_finding}")
    print(f"  - Single-label: {one_label}")
    print(f"  - Multi-label: {multi_label}")
    print("-" * 30)
    return no_finding, one_label, multi_label


def plot_co_occurrence_heatmap(csv_path, save_dir):
    """
    Generates and saves a heatmap of disease label co-occurrence from the training data.
    """
    print("\n--- Generating Co-occurrence Heatmap ---")
    df = pd.read_csv(csv_path)
    finding_list = df.columns[1:].tolist()
    num_classes = len(finding_list)
    
    label_array = df[finding_list].values
    co_mat = np.zeros((num_classes, num_classes), dtype=np.float32)

    for i in range(label_array.shape[0]):
        labels = np.where(label_array[i] == 1)[0]
        for l1 in labels:
            for l2 in labels:
                co_mat[l1, l2] += 1
    
    # Normalize by the count of each label (diagonal)
    co_mat_norm = co_mat / (np.diag(co_mat) + 1e-8)

    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(
        co_mat_norm,
        annot=True,
        xticklabels=finding_list,
        yticklabels=finding_list,
        annot_kws={"size": 9},
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5
    )

    plt.title("Normalized Disease Co-occurrence Heatmap (Train Set)", **TITLE_CONFIG)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "nih14_co_occurrence_heatmap.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved co-occurrence heatmap to: {save_path}")
    plt.close()


def analyze_bbox_distribution(bbox_csv_path, save_dir):
    """
    Analyzes and plots the distribution of diseases with bounding box annotations.
    """
    print("\n--- Analyzing BBox Distribution ---")
    bbox_df = pd.read_csv(bbox_csv_path)
    label_counts = Counter(bbox_df['Finding Label'])
    
    findings_list = sorted(label_counts.keys())
    num_list = [label_counts[disease] for disease in findings_list]

    print(f"Total images with BBox: {len(bbox_df)}")
    print(f"Disease counts with BBox: {dict(zip(findings_list, num_list))}")

    plt.figure(figsize=(12, 6))
    bars = sns.barplot(x=findings_list, y=num_list, edgecolor="k")

    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, int(yval),
                 ha="center", va="bottom", **FONT_CONFIG)

    plt.xticks(rotation=-20, ha='left')
    plt.xlabel("Disease Labels", **FONT_CONFIG)
    plt.ylabel("Number of BBoxes", **FONT_CONFIG)
    plt.title("Distribution of Diseases with Bounding Box Annotations", **TITLE_CONFIG)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "nih14_bbox_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved BBox distribution plot to: {save_path}")
    plt.close()


def main():
    """
    Main function to drive the analysis. It runs all analysis tasks
    with hardcoded paths for simplicity.
    """
    # Define paths relative to this script's location for robustness
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'files', 'nih14_label')
    save_dir = os.path.join(script_dir, 'plots')
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Reading data from: {data_dir}")
    print(f"Saving plots to: {save_dir}")

    # --- 1. Label Distribution and Sample Type Analysis ---
    print("\nRunning: Label Distribution and Sample Type Analysis")
    splits = ["train", "val", "test"]
    total_stats = {"no_finding": 0, "one_label": 0, "multi_label": 0, "total": 0}

    for split in splits:
        csv_path = os.path.join(data_dir, f"{split}_label.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping analysis for {split}.")
            continue
        
        df = pd.read_csv(csv_path)
        plot_label_distribution(df, split, save_dir)
        no_finding, one_label, multi_label = analyze_sample_types(df, split)
        
        total_stats["no_finding"] += no_finding
        total_stats["one_label"] += one_label
        total_stats["multi_label"] += multi_label
        total_stats["total"] += len(df)
        
    print("\n--- Overall Sample Type Summary ---")
    for key, value in total_stats.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")

    # --- 2. Co-occurrence Heatmap ---
    print("\nRunning: Co-occurrence Heatmap Analysis")
    train_csv_path = os.path.join(data_dir, "train_label.csv")
    if os.path.exists(train_csv_path):
        plot_co_occurrence_heatmap(train_csv_path, save_dir)
    else:
        print(f"Warning: {train_csv_path} not found. Skipping co-occurrence analysis.")

    # --- 3. BBox Distribution ---
    print("\nRunning: BBox Distribution Analysis")
    bbox_csv_path = os.path.join(data_dir, "BBox_List_2017.csv")
    if os.path.exists(bbox_csv_path):
        analyze_bbox_distribution(bbox_csv_path, save_dir)
    else:
        print(f"Warning: {bbox_csv_path} not found. Skipping BBox analysis.")
    
    print("\nAnalysis complete. All plots saved.")


if __name__ == "__main__":
    main() 