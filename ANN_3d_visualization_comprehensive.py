"""
Generates 3D visualizations comparing neural network model performance on MNIST digits 2,3,4.
Creates a 4×3 grid showing clean data vs FGSM attacks for BP, BL, and BL->BP methods
from three different viewing angles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
import sys

#------------------------------------------------------------------------------
# Configuration and Setup
#------------------------------------------------------------------------------
# Base configuration
base_path = os.path.dirname(os.path.abspath(__file__))
print(f"Using local path: {base_path}")

# Model configurations
configs = {
    "BP": {
        "folder_csv": "mnist_nn_nobias_backprop",
        "folder_weights": "nobias_backprop"
    },
    "BL": {
        "folder_csv": "mnist_nn_nobias_biprop",
        "folder_weights": "nobias_biprop"
    },
    "BL->BP": {
        "folder_csv": "mnist_nn_nobias_halfbiprop",
        "folder_weights": "nobias_halfbiprop"
    }
}

# Visualization colors
cmap_points = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])  # Red, Green, Blue for digits 2,3,4
LABEL_TO_IDX = {2: 0, 3: 1, 4: 2}
BASE_COLORS_3 = [
    (1.0, 0.0, 0.0),  # Red for digit 2
    (0.0, 1.0, 0.0),  # Green for digit 3
    (0.0, 0.0, 1.0)   # Blue for digit 4
]

#------------------------------------------------------------------------------
# Data Loading and Processing Functions
#------------------------------------------------------------------------------
def parse_summary_best_block(summary_txt_path):
    # Extract best accuracy values from summary file
    best_clean = best_noise = best_fgsm = 0.0
    
    if not os.path.exists(summary_txt_path):
        return best_clean, best_noise, best_fgsm
    
    with open(summary_txt_path, "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip().lower()
        if line.startswith("best accuracy test data:"):
            j = i + 1
            while j < len(lines):
                l2 = lines[j].strip().lower()
                if l2.startswith("best accuracy") and not l2.startswith("best accuracy test data:"):
                    break
                if l2.startswith("test data:"):
                    val_str = lines[j].split(":")[1].strip()
                    if val_str:
                        best_clean = float(val_str)
                elif l2.startswith("test data with noise:"):
                    val_str = lines[j].split(":")[1].strip()
                    if val_str:
                        best_noise = float(val_str)
                elif l2.startswith("test data with fgsm:"):
                    val_str = lines[j].split(":")[1].strip()
                    if val_str:
                        best_fgsm = float(val_str)
                j += 1
            break
        i += 1
    
    return best_clean, best_noise, best_fgsm

def load_config_data(config_name):
    # Load data and weights for given configuration
    folder_csv = configs[config_name]["folder_csv"]
    
    # Load accuracy values
    summary_path = os.path.join(base_path, "csv", folder_csv, "ANN_summary.txt")
    best_clean, best_noise, best_fgsm = parse_summary_best_block(summary_path)
    
    # Load test data
    path_clean = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_images.npy")
    path_noise = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_noise.npy")
    path_fgsm = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_fgsm.npy")
    path_labels = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_labels.npy")
    
    # Load data files
    X_clean = np.load(path_clean)
    X_noise = np.load(path_noise)
    X_fgsm = np.load(path_fgsm)
    y_234 = np.load(path_labels)
    
    # Load model weights
    w_path = os.path.join(base_path, "Weights", configs[config_name]["folder_weights"], "ANN_C_W1.npy")
    b_path = os.path.join(base_path, "Weights", configs[config_name]["folder_weights"], "ANN_C_B1.npy")
    
    W = np.load(w_path)
    if os.path.exists(b_path):
        b = np.load(b_path)
    else:
        print(f"Using zero bias for {config_name} (nobias model)")
        b = np.zeros(W.shape[1])
    
    return {
        "Clean": (X_clean, best_clean),
        "Noise": (X_noise, best_noise),
        "FGSM": (X_fgsm, best_fgsm)
    }, y_234, W, b

def filter_label_234(X, y):
    # Keep only samples with labels 2,3,4
    mask = np.isin(y, [2, 3, 4])
    return X[mask], y[mask]

#------------------------------------------------------------------------------
# Visualization Functions
#------------------------------------------------------------------------------
def create_clean_vs_fgsm_comparison():
    # Create 4×3 grid visualization comparing clean data vs FGSM attacks
    vis_dir = os.path.join(base_path, "visualization_3d")
    os.makedirs(vis_dir, exist_ok=True)
    
    methods = ["BP", "BL", "BL->BP"]
    
    # Set up figure layout
    fig = plt.figure(figsize=(30, 24))
    plt.subplots_adjust(
        left=0.15, right=0.73, top=0.90, bottom=0.08,
        wspace=-0.01, hspace=0.05
    )
    
    # Configure labels
    row_labels = ["Clean", "BP FGSM", "BL FGSM", "BL→BP FGSM"]
    col_labels = ["View 1", "View 2", "View 3"]
    
    # Add column headers
    total_width = 0.83 - 0.15
    column_width = total_width / 3
    for col_idx in range(3):
        x_pos = 0.15 + column_width * (col_idx + 0.5)
        fig.text(x_pos, 0.93,
                col_labels[col_idx],
                ha='center',
                va='center',
                fontsize=22,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=5))
    
    # Load data for all methods
    data_bp, y_bp, _, _ = load_config_data("BP")
    data_bl, y_bl, _, _ = load_config_data("BL")
    data_blbp, y_blbp, _, _ = load_config_data("BL->BP")
    
    # Prepare PCA transformation
    X_clean_bp, _ = data_bp["Clean"]
    X_filt_bp, y_filt_bp = filter_label_234(X_clean_bp, y_bp)
    pca_shared = PCA(n_components=3, random_state=42)
    pca_shared.fit(X_filt_bp)
    
    # Print variance explained
    explained_variance = pca_shared.explained_variance_ratio_
    total_variance = sum(explained_variance) * 100
    print("\nPCA Explained Variance:")
    print(f"PC1: {explained_variance[0]*100:.1f}%")
    print(f"PC2: {explained_variance[1]*100:.1f}%")
    print(f"PC3: {explained_variance[2]*100:.1f}%")
    print(f"Total variance explained: {total_variance:.1f}%")
    
    # Define viewing angles
    views = [
        {'elev': 30, 'azim': 45},  # View 1
        {'elev': 0, 'azim': 0},    # View 2
        {'elev': 90, 'azim': 0}    # View 3
    ]
    
    # Plot clean data (row 1)
    X_data, acc = data_bp["Clean"]
    X_filt, y_filt = filter_label_234(X_data, y_bp)
    X_3D = pca_shared.transform(X_filt)
    
    colors = ['red', 'green', 'blue']
    class_map = {2: 0, 3: 1, 4: 2}
    
    for col_idx in range(3):
        ax = fig.add_subplot(4, 3, col_idx + 1, projection='3d')
        
        # Plot data points
        c = [colors[class_map[label]] for label in y_filt]
        ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2],
                  c=c, alpha=0.6, marker='o', s=30)
        
        ax.view_init(elev=views[col_idx]['elev'], azim=views[col_idx]['azim'])
        
        # Configure axes for each view
        if col_idx == 0:
            ax.set_xlabel(f'PC1', fontsize=25, labelpad=32)
            ax.set_ylabel(f'PC2', fontsize=25, labelpad=30)
            ax.set_zlabel(f'PC3', fontsize=25, labelpad=25)
            ax.tick_params(axis='x', labelsize=18, pad=10)
            ax.tick_params(axis='y', labelsize=18, pad=10)
            ax.tick_params(axis='z', labelsize=18, pad=10)
        elif col_idx == 1:
            ax.set_xlabel(f'PC1', fontsize=25, labelpad=32)
            ax.set_ylabel(f'PC2', fontsize=25, labelpad=30)
            ax.set_zlabel(f'PC3', fontsize=25, labelpad=25)
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelsize=18, pad=10)
            ax.tick_params(axis='z', labelsize=18, pad=10)
        else:
            ax.set_xlabel(f'PC1', fontsize=25, labelpad=32)
            ax.set_ylabel(f'PC2', fontsize=25, labelpad=30)
            ax.set_zlabel(f'PC3', fontsize=25, labelpad=25)
            ax.tick_params(axis='x', labelsize=18, pad=10)
            ax.tick_params(axis='y', labelsize=18, pad=10)
            ax.tick_params(axis='z', labelleft=False)
        
        # Add row label and accuracy
        if col_idx == 0:
            ax.text2D(-0.45, 0.5, f"{row_labels[0]}\nAcc: {acc*100:.1f}%",
                     transform=ax.transAxes,
                     rotation=0,
                     fontsize=20,
                     fontweight='bold',
                     verticalalignment='center',
                     horizontalalignment='center',
                     bbox=dict(facecolor='white', 
                              alpha=0.8, 
                              edgecolor='black', 
                              pad=3,
                              boxstyle='round,pad=0.5'))
    
    # Plot FGSM data (rows 2-4)
    for row_idx, method in enumerate(methods, 1):
        # Get data for current method
        if method == "BP":
            X_data, acc = data_bp["FGSM"]
            y_labels = y_bp
        elif method == "BL":
            X_data, acc = data_bl["FGSM"]
            y_labels = y_bl
        else:
            X_data, acc = data_blbp["FGSM"]
            y_labels = y_blbp
        
        X_filt, y_filt = filter_label_234(X_data, y_labels)
        X_3D = pca_shared.transform(X_filt)
        
        for col_idx in range(3):
            ax = fig.add_subplot(4, 3, (row_idx * 3) + col_idx + 1, projection='3d')
            
            # Plot data points
            c = [colors[class_map[label]] for label in y_filt]
            ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2],
                      c=c, alpha=0.6, marker='o', s=30)
            
            ax.view_init(elev=views[col_idx]['elev'], azim=views[col_idx]['azim'])
            
            # Configure axes
            ax.set_xlabel('', fontsize=20)
            ax.set_ylabel('', fontsize=20)
            ax.set_zlabel('', fontsize=20)
            
            if col_idx == 0:
                ax.tick_params(axis='x', labelsize=18, pad=10)
                ax.tick_params(axis='y', labelsize=18, pad=10)
                ax.tick_params(axis='z', labelsize=18, pad=10)
            elif col_idx == 1:
                ax.tick_params(axis='x', labelbottom=False)
                ax.tick_params(axis='y', labelsize=18, pad=10)
                ax.tick_params(axis='z', labelsize=18, pad=10)
            else:
                ax.tick_params(axis='x', labelsize=18, pad=10)
                ax.tick_params(axis='y', labelsize=18, pad=10)
                ax.tick_params(axis='z', labelleft=False)
            
            # Add row label and accuracy
            if col_idx == 0:
                ax.text2D(-0.45, 0.5, f"{row_labels[row_idx]}\nAcc: {acc*100:.1f}%",
                         transform=ax.transAxes,
                         rotation=0,
                         fontsize=22,
                         fontweight='bold',
                         verticalalignment='center',
                         horizontalalignment='center',
                         bbox=dict(facecolor='white', 
                                  alpha=0.8, 
                                  edgecolor='black', 
                                  pad=3,
                                  boxstyle='round,pad=0.5'))
    
    # Add legend
    legend_elements = [
        Patch(facecolor='red', label='Digit 2'),
        Patch(facecolor='green', label='Digit 3'),
        Patch(facecolor='blue', label='Digit 4'),
    ]
    
    fig.legend(handles=legend_elements, 
              loc='lower center',
              ncol=3,
              bbox_to_anchor=(0.45, 0.02),
              fontsize=24,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # Save figure
    save_path = os.path.join(vis_dir, "17-clean_vs_fgsm_all_methods_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined clean vs FGSM comparison to: {save_path}")

#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------
def main_essential_visualization():
    # Create visualization comparing clean data vs FGSM attacks
    methods = ["BP", "BL", "BL->BP"]
    
    vis_dir = os.path.join(base_path, "visualization_3d")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Using base path: {base_path}")
    print(f"Output directory: {vis_dir}")
    
    # Prepare PCA transformation
    print("\nLoading BP data and fitting shared PCA...")
    data_bp, y_bp, W_bp, b_bp = load_config_data("BP")
    X_clean_bp, _ = data_bp["Clean"]
    X_filt_bp, y_filt_bp = filter_label_234(X_clean_bp, y_bp)
    
    pca_shared = PCA(n_components=3, random_state=42)
    pca_shared.fit(X_filt_bp)
    explained_variance = pca_shared.explained_variance_ratio_
    total_variance = sum(explained_variance) * 100

    print("\nPCA Explained Variance:")
    print(f"PC1: {explained_variance[0]*100:.1f}%")
    print(f"PC2: {explained_variance[1]*100:.1f}%")
    print(f"PC3: {explained_variance[2]*100:.1f}%")
    print(f"Total variance explained: {total_variance:.1f}%")
    
    # Generate visualization
    print("\nCreating Clean vs FGSM comparison visualization...")
    create_clean_vs_fgsm_comparison()
    
    print(f"\nVisualization has been saved to: {os.path.join(vis_dir, '17-clean_vs_fgsm_all_methods_comparison.png')}")

if __name__ == "__main__":
    main_essential_visualization()