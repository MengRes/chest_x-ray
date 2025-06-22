import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CheXpertAnalyzer:
    def __init__(self, data_dir="/home/mxz3935/dataset_folder/chexpert_v1.0_small"):
        self.data_dir = data_dir
        self.train_csv = os.path.join(data_dir, "train.csv")
        self.valid_csv = os.path.join(data_dir, "valid.csv")
        
        # CheXpert classes
        self.classes = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # Load data
        self.train_df = None
        self.valid_df = None
        self.load_data()
    
    def load_data(self):
        """Load train and validation data"""
        print("Loading CheXpert dataset...")
        
        if os.path.exists(self.train_csv):
            self.train_df = pd.read_csv(self.train_csv)
            print(f"Train data loaded: {len(self.train_df)} samples")
        else:
            print(f"Train CSV not found: {self.train_csv}")
            
        if os.path.exists(self.valid_csv):
            self.valid_df = pd.read_csv(self.valid_csv)
            print(f"Validation data loaded: {len(self.valid_df)} samples")
        else:
            print(f"Validation CSV not found: {self.valid_csv}")
    
    def process_labels(self, df, policy="zeros"):
        """Process labels according to policy"""
        if df is None:
            return None
            
        # Get label columns (from column 5 onwards)
        label_cols = df.columns[5:5+len(self.classes)]
        
        processed_labels = []
        for _, row in df.iterrows():
            labels = []
            for col in label_cols:
                label_val = row[col]
                
                if pd.isna(label_val) or label_val == '':
                    labels.append(0)
                else:
                    label_val = float(label_val)
                    if label_val == 1:
                        labels.append(1)
                    elif label_val == -1:
                        labels.append(1 if policy == "ones" else 0)
                    else:
                        labels.append(0)
            
            processed_labels.append(labels)
        
        return np.array(processed_labels)
    
    def analyze_label_distribution(self, policy="zeros"):
        """Analyze label distribution for both train and validation sets"""
        print(f"\nAnalyzing label distribution with policy: {policy}")
        
        # Process labels
        train_labels = self.process_labels(self.train_df, policy)
        valid_labels = self.process_labels(self.valid_df, policy)
        
        results = {}
        
        if train_labels is not None:
            train_counts = np.sum(train_labels, axis=0)
            train_percentages = (train_counts / len(train_labels)) * 100
            results['train'] = {
                'counts': train_counts,
                'percentages': train_percentages,
                'total_samples': len(train_labels)
            }
        
        if valid_labels is not None:
            valid_counts = np.sum(valid_labels, axis=0)
            valid_percentages = (valid_counts / len(valid_labels)) * 100
            results['valid'] = {
                'counts': valid_counts,
                'percentages': valid_percentages,
                'total_samples': len(valid_labels)
            }
        
        return results
    
    def create_summary_table(self, results):
        """Create a summary table of label distribution"""
        print("\n" + "="*80)
        print("CHEXPERT DATASET LABEL DISTRIBUTION SUMMARY")
        print("="*80)
        
        # Create DataFrame for better display
        summary_data = []
        
        for i, class_name in enumerate(self.classes):
            row = {'Class': class_name}
            
            if 'train' in results:
                row['Train_Count'] = results['train']['counts'][i]
                row['Train_Percentage'] = f"{results['train']['percentages'][i]:.2f}%"
            
            if 'valid' in results:
                row['Valid_Count'] = results['valid']['counts'][i]
                row['Valid_Percentage'] = f"{results['valid']['percentages'][i]:.2f}%"
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv('label_distribution_summary.csv', index=False)
        print(f"\nSummary saved to: label_distribution_summary.csv")
        
        return summary_df
    
    def plot_label_distribution(self, results, policy="zeros"):
        """Create visualization of label distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'CheXpert Dataset Label Distribution (Policy: {policy})', fontsize=16, fontweight='bold')
        
        # Train set counts
        if 'train' in results:
            ax1 = axes[0, 0]
            counts = results['train']['counts']
            ax1.bar(range(len(self.classes)), counts, color='skyblue', alpha=0.7)
            ax1.set_title('Train Set - Label Counts')
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            ax1.set_xticks(range(len(self.classes)))
            ax1.set_xticklabels(self.classes, ha='right')
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                ax1.text(i, count + max(counts)*0.01, f'{count:,}', 
                        ha='center', va='bottom', fontsize=8)
        
        # Train set percentages
        if 'train' in results:
            ax2 = axes[0, 1]
            percentages = results['train']['percentages']
            ax2.bar(range(len(self.classes)), percentages, color='lightgreen', alpha=0.7)
            ax2.set_title('Train Set - Label Percentages')
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Percentage (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_xticks(range(len(self.classes)))
            ax2.set_xticklabels(self.classes, ha='right')
            
            # Add percentage labels on bars
            for i, pct in enumerate(percentages):
                ax2.text(i, pct + max(percentages)*0.01, f'{pct:.1f}%', 
                        ha='center', va='bottom', fontsize=8)
        
        # Validation set counts
        if 'valid' in results:
            ax3 = axes[1, 0]
            counts = results['valid']['counts']
            ax3.bar(range(len(self.classes)), counts, color='salmon', alpha=0.7)
            ax3.set_title('Validation Set - Label Counts')
            ax3.set_xlabel('Classes')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)
            ax3.set_xticks(range(len(self.classes)))
            ax3.set_xticklabels(self.classes, ha='right')
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                ax3.text(i, count + max(counts)*0.01, f'{count:,}', 
                        ha='center', va='bottom', fontsize=8)
        
        # Validation set percentages
        if 'valid' in results:
            ax4 = axes[1, 1]
            percentages = results['valid']['percentages']
            ax4.bar(range(len(self.classes)), percentages, color='gold', alpha=0.7)
            ax4.set_title('Validation Set - Label Percentages')
            ax4.set_xlabel('Classes')
            ax4.set_ylabel('Percentage (%)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.set_xticks(range(len(self.classes)))
            ax4.set_xticklabels(self.classes, ha='right')
            
            # Add percentage labels on bars
            for i, pct in enumerate(percentages):
                ax4.text(i, pct + max(percentages)*0.01, f'{pct:.1f}%', 
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('plots/label_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: plots/label_distribution.png")
        plt.show()
    
    def analyze_uncertain_labels(self):
        """Analyze uncertain labels (-1 values)"""
        print("\n" + "="*50)
        print("UNCERTAIN LABELS ANALYSIS (-1 values)")
        print("="*50)
        
        uncertain_stats = {}
        
        for dataset_name, df in [('Train', self.train_df), ('Validation', self.valid_df)]:
            if df is None:
                continue
                
            label_cols = df.columns[5:5+len(self.classes)]
            uncertain_counts = []
            
            for col in label_cols:
                uncertain_count = (df[col] == -1).sum()
                uncertain_counts.append(uncertain_count)
            
            uncertain_stats[dataset_name] = uncertain_counts
            
            print(f"\n{dataset_name} Set Uncertain Labels:")
            for i, class_name in enumerate(self.classes):
                count = uncertain_counts[i]
                percentage = (count / len(df)) * 100
                print(f"  {class_name}: {count:,} ({percentage:.2f}%)")
        
        # Create uncertain labels plot
        self.plot_uncertain_labels(uncertain_stats)
        
        return uncertain_stats
    
    def plot_uncertain_labels(self, uncertain_stats):
        """Plot uncertain labels distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('CheXpert Dataset - Uncertain Labels (-1) Distribution', fontsize=16, fontweight='bold')
        
        for idx, (dataset_name, counts) in enumerate(uncertain_stats.items()):
            ax = axes[idx]
            colors = ['red' if count > 0 else 'lightgray' for count in counts]
            
            bars = ax.bar(range(len(self.classes)), counts, color=colors, alpha=0.7)
            ax.set_title(f'{dataset_name} Set')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Count of Uncertain Labels')
            ax.tick_params(axis='x', rotation=45)
            ax.set_xticks(range(len(self.classes)))
            ax.set_xticklabels(self.classes, ha='right')
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                if count > 0:
                    ax.text(i, count + max(counts)*0.01, f'{count:,}', 
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('plots/uncertain_labels.png', dpi=300, bbox_inches='tight')
        print(f"Uncertain labels plot saved to: plots/uncertain_labels.png")
        plt.show()
    
    def analyze_demographics(self):
        """Analyze demographic information"""
        print("\n" + "="*50)
        print("DEMOGRAPHIC ANALYSIS")
        print("="*50)
        
        for dataset_name, df in [('Train', self.train_df), ('Validation', self.valid_df)]:
            if df is None:
                continue
                
            print(f"\n{dataset_name} Set Demographics:")
            
            # Sex distribution
            sex_counts = df['Sex'].value_counts()
            print(f"  Sex Distribution:")
            for sex, count in sex_counts.items():
                percentage = (count / len(df)) * 100
                print(f"    {sex}: {count:,} ({percentage:.2f}%)")
            
            # Age distribution
            age_stats = df['Age'].describe()
            print(f"  Age Statistics:")
            print(f"    Mean: {age_stats['mean']:.1f}")
            print(f"    Median: {age_stats['50%']:.1f}")
            print(f"    Min: {age_stats['min']:.1f}")
            print(f"    Max: {age_stats['max']:.1f}")
            
            # View type distribution
            view_counts = df['Frontal/Lateral'].value_counts()
            print(f"  View Type Distribution:")
            for view, count in view_counts.items():
                percentage = (count / len(df)) * 100
                print(f"    {view}: {count:,} ({percentage:.2f}%)")
            
            # AP/PA distribution
            ap_pa_counts = df['AP/PA'].value_counts()
            print(f"  AP/PA Distribution:")
            for ap_pa, count in ap_pa_counts.items():
                percentage = (count / len(df)) * 100
                print(f"    {ap_pa}: {count:,} ({percentage:.2f}%)")
    
    def run_complete_analysis(self):
        """Run complete analysis"""
        print("Starting CheXpert Dataset Analysis...")
        print("="*60)
        
        # Basic dataset info
        print(f"Dataset location: {self.data_dir}")
        print(f"Number of classes: {len(self.classes)}")
        
        # Analyze label distribution for both policies
        for policy in ["zeros", "ones"]:
            results = self.analyze_label_distribution(policy)
            summary_df = self.create_summary_table(results)
            self.plot_label_distribution(results, policy)
        
        # Analyze uncertain labels
        uncertain_stats = self.analyze_uncertain_labels()
        
        # Analyze label value distribution
        label_value_stats = self.analyze_label_value_distribution()
        
        # Analyze demographics
        self.analyze_demographics()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  - label_distribution_summary.csv")
        print("  - label_value_distribution_summary.csv")
        print("  - plots/label_distribution.png")
        print("  - plots/uncertain_labels.png")
        print("  - plots/label_value_distribution.png")
        print("  - plots/label_value_percentage_distribution.png")

    def analyze_label_value_distribution(self):
        """Analyze the distribution of label values (0, 1, -1) for each disease class"""
        print("\n" + "="*80)
        print("LABEL VALUE DISTRIBUTION ANALYSIS (0, 1, -1)")
        print("="*80)
        
        label_value_stats = {}
        
        for dataset_name, df in [('Train', self.train_df), ('Validation', self.valid_df)]:
            if df is None:
                continue
                
            print(f"\n{dataset_name} Set Label Value Distribution:")
            print("-" * 80)
            
            # Get label columns (from column 5 onwards)
            label_cols = df.columns[5:5+len(self.classes)]
            
            dataset_stats = {}
            
            for i, (class_name, col) in enumerate(zip(self.classes, label_cols)):
                # Count each label value
                value_counts = df[col].value_counts()
                
                # Initialize counts
                count_0 = value_counts.get(0, 0)
                count_1 = value_counts.get(1, 0)
                count_minus_1 = value_counts.get(-1, 0)
                count_na = df[col].isna().sum()
                
                # Calculate percentages
                total = len(df)
                pct_0 = (count_0 / total) * 100
                pct_1 = (count_1 / total) * 100
                pct_minus_1 = (count_minus_1 / total) * 100
                pct_na = (count_na / total) * 100
                
                dataset_stats[class_name] = {
                    'count_0': count_0,
                    'count_1': count_1,
                    'count_minus_1': count_minus_1,
                    'count_na': count_na,
                    'pct_0': pct_0,
                    'pct_1': pct_1,
                    'pct_minus_1': pct_minus_1,
                    'pct_na': pct_na
                }
                
                print(f"{class_name:25} | 0: {count_0:6,} ({pct_0:5.2f}%) | 1: {count_1:6,} ({pct_1:5.2f}%) | -1: {count_minus_1:6,} ({pct_minus_1:5.2f}%) | NA: {count_na:4,} ({pct_na:5.2f}%)")
            
            label_value_stats[dataset_name] = dataset_stats
        
        # Create summary DataFrame
        self.create_label_value_summary_table(label_value_stats)
        
        # Create visualization
        self.plot_label_value_distribution(label_value_stats)
        
        return label_value_stats
    
    def create_label_value_summary_table(self, label_value_stats):
        """Create a summary table of label value distribution"""
        print("\n" + "="*100)
        print("LABEL VALUE DISTRIBUTION SUMMARY TABLE")
        print("="*100)
        
        # Create DataFrame for better display
        summary_data = []
        
        for class_name in self.classes:
            row = {'Class': class_name}
            
            for dataset_name in ['Train', 'Validation']:
                if dataset_name in label_value_stats:
                    stats = label_value_stats[dataset_name][class_name]
                    row[f'{dataset_name}_0_Count'] = stats['count_0']
                    row[f'{dataset_name}_0_Pct'] = f"{stats['pct_0']:.2f}%"
                    row[f'{dataset_name}_1_Count'] = stats['count_1']
                    row[f'{dataset_name}_1_Pct'] = f"{stats['pct_1']:.2f}%"
                    row[f'{dataset_name}_-1_Count'] = stats['count_minus_1']
                    row[f'{dataset_name}_-1_Pct'] = f"{stats['pct_minus_1']:.2f}%"
                    row[f'{dataset_name}_NA_Count'] = stats['count_na']
                    row[f'{dataset_name}_NA_Pct'] = f"{stats['pct_na']:.2f}%"
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display with better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv('label_value_distribution_summary.csv', index=False)
        print(f"\nLabel value distribution summary saved to: label_value_distribution_summary.csv")
        
        return summary_df
    
    def plot_label_value_distribution(self, label_value_stats):
        """Create visualization of label value distribution"""
        fig, axes = plt.subplots(2, 1, figsize=(20, 16))
        fig.suptitle('CheXpert Dataset - Label Value Distribution (0, 1, -1, NA)', fontsize=16, fontweight='bold')
        
        colors = ['lightblue', 'lightgreen', 'salmon', 'lightgray']
        labels = ['0 (Negative)', '1 (Positive)', '-1 (Uncertain)', 'NA (Missing)']
        
        for idx, dataset_name in enumerate(['Train', 'Validation']):
            if dataset_name not in label_value_stats:
                continue
                
            ax = axes[idx]
            dataset_stats = label_value_stats[dataset_name]
            
            # Prepare data for stacked bar chart
            x = range(len(self.classes))
            y_0 = [dataset_stats[class_name]['count_0'] for class_name in self.classes]
            y_1 = [dataset_stats[class_name]['count_1'] for class_name in self.classes]
            y_minus_1 = [dataset_stats[class_name]['count_minus_1'] for class_name in self.classes]
            y_na = [dataset_stats[class_name]['count_na'] for class_name in self.classes]
            
            # Create stacked bar chart
            bars = ax.bar(x, y_0, label=labels[0], color=colors[0], alpha=0.8)
            ax.bar(x, y_1, bottom=y_0, label=labels[1], color=colors[1], alpha=0.8)
            ax.bar(x, y_minus_1, bottom=[i+j for i,j in zip(y_0, y_1)], label=labels[2], color=colors[2], alpha=0.8)
            ax.bar(x, y_na, bottom=[i+j+k for i,j,k in zip(y_0, y_1, y_minus_1)], label=labels[3], color=colors[3], alpha=0.8)
            
            ax.set_title(f'{dataset_name} Set - Label Value Distribution')
            ax.set_xlabel('Disease Classes')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            ax.set_xticks(x)
            ax.set_xticklabels(self.classes, ha='right')
            ax.legend()
            
            # Add total count labels on top of bars
            totals = [sum([y_0[i], y_1[i], y_minus_1[i], y_na[i]]) for i in range(len(self.classes))]
            for i, total in enumerate(totals):
                ax.text(i, total + max(totals)*0.01, f'{total:,}', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/label_value_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Label value distribution plot saved to: plots/label_value_distribution.png")
        plt.show()
        
        # Create percentage plot
        self.plot_label_value_percentage_distribution(label_value_stats)
    
    def plot_label_value_percentage_distribution(self, label_value_stats):
        """Create percentage visualization of label value distribution"""
        fig, axes = plt.subplots(2, 1, figsize=(20, 16))
        fig.suptitle('CheXpert Dataset - Label Value Percentage Distribution', fontsize=16, fontweight='bold')
        
        colors = ['lightblue', 'lightgreen', 'salmon', 'lightgray']
        labels = ['0 (Negative)', '1 (Positive)', '-1 (Uncertain)', 'NA (Missing)']
        
        for idx, dataset_name in enumerate(['Train', 'Validation']):
            if dataset_name not in label_value_stats:
                continue
                
            ax = axes[idx]
            dataset_stats = label_value_stats[dataset_name]
            
            # Prepare percentage data for stacked bar chart
            x = range(len(self.classes))
            pct_0 = [dataset_stats[class_name]['pct_0'] for class_name in self.classes]
            pct_1 = [dataset_stats[class_name]['pct_1'] for class_name in self.classes]
            pct_minus_1 = [dataset_stats[class_name]['pct_minus_1'] for class_name in self.classes]
            pct_na = [dataset_stats[class_name]['pct_na'] for class_name in self.classes]
            
            # Create stacked bar chart
            bars = ax.bar(x, pct_0, label=labels[0], color=colors[0], alpha=0.8)
            ax.bar(x, pct_1, bottom=pct_0, label=labels[1], color=colors[1], alpha=0.8)
            ax.bar(x, pct_minus_1, bottom=[i+j for i,j in zip(pct_0, pct_1)], label=labels[2], color=colors[2], alpha=0.8)
            ax.bar(x, pct_na, bottom=[i+j+k for i,j,k in zip(pct_0, pct_1, pct_minus_1)], label=labels[3], color=colors[3], alpha=0.8)
            
            ax.set_title(f'{dataset_name} Set - Label Value Percentage Distribution')
            ax.set_xlabel('Disease Classes')
            ax.set_ylabel('Percentage (%)')
            ax.tick_params(axis='x', rotation=45)
            ax.set_xticks(x)
            ax.set_xticklabels(self.classes, ha='right')
            ax.legend()
            ax.set_ylim(0, 100)
            
            # Add percentage labels on bars
            for i in range(len(self.classes)):
                # Add labels for each segment
                if pct_0[i] > 5:  # Only show label if segment is large enough
                    ax.text(i, pct_0[i]/2, f'{pct_0[i]:.1f}%', ha='center', va='center', fontsize=8)
                if pct_1[i] > 5:
                    ax.text(i, pct_0[i] + pct_1[i]/2, f'{pct_1[i]:.1f}%', ha='center', va='center', fontsize=8)
                if pct_minus_1[i] > 5:
                    ax.text(i, pct_0[i] + pct_1[i] + pct_minus_1[i]/2, f'{pct_minus_1[i]:.1f}%', ha='center', va='center', fontsize=8)
                if pct_na[i] > 5:
                    ax.text(i, pct_0[i] + pct_1[i] + pct_minus_1[i] + pct_na[i]/2, f'{pct_na[i]:.1f}%', ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('plots/label_value_percentage_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Label value percentage distribution plot saved to: plots/label_value_percentage_distribution.png")
        plt.show()

if __name__ == "__main__":
    # Create analyzer and run analysis
    analyzer = CheXpertAnalyzer()
    analyzer.run_complete_analysis() 