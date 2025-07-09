"""
ANN_2D_visual.py

Visualizes the decision boundaries and data distributions of different neural network
training methods (BP, BL, BL->BP) and classical models (SVM, GP) in 2D space.
Uses PCA for dimensionality reduction and compares model performance on clean,
noisy, and FGSM-perturbed MNIST data (digits 2,3,4).
"""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
import os
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()
print("TensorFlow version:", tf.__version__)

#------------------------------------------------------------------------------
# Configuration and Directory Setup
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

# Verify paths
for config_name, config in configs.items():
    folder_csv = config["folder_csv"]
    folder_weights = config["folder_weights"]
    
    csv_path = os.path.join(base_path, "csv", folder_csv)
    weights_path = os.path.join(base_path, "Weights", folder_weights)
    
    print(f"Checking paths for {config_name}:")
    print(f"  CSV path: {csv_path} - {'EXISTS' if os.path.exists(csv_path) else 'MISSING'}")
    print(f"  Weights path: {weights_path} - {'EXISTS' if os.path.exists(weights_path) else 'MISSING'}")

#------------------------------------------------------------------------------
# Visualization Configuration
#------------------------------------------------------------------------------
# Color mapping for visualization
cmap_points = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])  # Red, Green, Blue for digits 2,3,4
LABEL_TO_IDX = {2:0, 3:1, 4:2}
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
    best_clean = 0.0
    best_noise = 0.0
    best_fgsm  = 0.0

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
    # Load model data and weights for given configuration
    print(f"Loading {config_name} data...")
    
    folder_csv = configs[config_name]["folder_csv"]
    folder_w   = configs[config_name]["folder_weights"]

    # Get accuracy values from summary
    summary_path = os.path.join(base_path, "csv", folder_csv, "ANN_summary.txt")
    best_clean, best_noise, best_fgsm = parse_summary_best_block(summary_path)

    # Load test data files
    path_clean  = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_images.npy")
    path_noise  = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_noise.npy")
    path_fgsm   = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_fgsm.npy")
    path_labels = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_labels.npy")

    # Check data files
    for path, name in [(path_clean, "clean images"), (path_noise, "noisy images"), 
                       (path_fgsm, "FGSM images"), (path_labels, "labels")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found at {path}")
            sys.exit(1)

    # Load data
    X_clean = np.load(path_clean)
    X_noise = np.load(path_noise)
    X_fgsm  = np.load(path_fgsm)
    y_234   = np.load(path_labels)

    # Load weights
    w_path = os.path.join(base_path, "Weights", folder_w, "ANN_C_W1.npy")
    b_path = os.path.join(base_path, "Weights", folder_w, "ANN_C_B1.npy")
    
    W = np.load(w_path)
    if os.path.exists(b_path):
        b = np.load(b_path)
    else:
        print(f"Using zero bias for {config_name} (nobias model)")
        b = np.zeros(W.shape[1])

    print("DEBUG:", config_name, "W.shape =", W.shape, "b.shape =", b.shape)

    return {
        "Clean": (X_clean, best_clean),
        "Noise": (X_noise, best_noise),
        "FGSM":  (X_fgsm,  best_fgsm)
    }, y_234, W, b

#------------------------------------------------------------------------------
# Model Prediction Functions
#------------------------------------------------------------------------------
def predict_custom_784_proba_234(X_784, W, b):
    # Forward pass through network for digits 2,3,4
    logits_10 = X_784 @ W + b
    shift = logits_10 - np.max(logits_10, axis=1, keepdims=True)
    exp_vals = np.exp(shift)
    sums_10 = np.sum(exp_vals, axis=1, keepdims=True)
    probs_10 = exp_vals / sums_10

    # Get probabilities for digits 2,3,4
    partial = probs_10[:, [2,3,4]]
    sums_234 = np.sum(partial, axis=1, keepdims=True)
    probs_3 = partial / sums_234
    return probs_3

def filter_label_234(X, y):
    # Keep only samples with labels 2,3,4
    mask = np.isin(y, [2,3,4])
    return X[mask], y[mask]

def get_interval_color_3(class_idx, p):
    # Get color intensity based on probability
    if p < 0.2:
        return (1.0, 1.0, 1.0)  # White for low probability

    base_r, base_g, base_b = BASE_COLORS_3[class_idx]
    
    # Set color intensity
    if p < 0.4:
        factor = 0.3
    elif p < 0.6:
        factor = 0.5
    elif p < 0.8:
        factor = 0.7
    else:
        factor = 0.9

    # Mix white and base color
    r = (1.0 - factor) + factor * base_r
    g = (1.0 - factor) + factor * base_g
    b = (1.0 - factor) + factor * base_b
    return (r, g, b)

#------------------------------------------------------------------------------
# Visualization Functions
#------------------------------------------------------------------------------
def plot_sub_manual_pca_layered(ax, X_784, y, pca, x_min, x_max, y_min, y_max,
                               clf_sklearn=None, W=None, b=None, best_acc=None, 
                               show_axes=False):
    # Transform data to 2D
    X_filt, y_filt = filter_label_234(X_784, y)
    X_2D = pca.transform(X_filt)

    # Calculate clustering quality
    try:
        sil_score = silhouette_score(X_2D, y_filt)
    except:
        sil_score = float('nan')
    
    # Create visualization grid
    h = 0.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid_2D = np.c_[xx.ravel(), yy.ravel()]
    grid_784 = pca.inverse_transform(grid_2D)

    # Get predictions
    if clf_sklearn is not None:
        Z_probs = clf_sklearn.predict_proba(grid_784)
        score_val = clf_sklearn.score(X_filt, y_filt)
        classes_ = clf_sklearn.classes_
    else:
        Z_probs = predict_custom_784_proba_234(grid_784, W, b)
        score_val = best_acc
        classes_ = np.array([2,3,4])

    # Create decision boundary colors
    Z_colors = np.zeros((len(grid_2D), 3), dtype=np.float32)
    for i in range(len(grid_2D)):
        row_probs = Z_probs[i]
        c_ = np.argmax(row_probs)
        p = row_probs[c_]
        label_win = classes_[c_]
        idx_color = LABEL_TO_IDX[label_win]
        Z_colors[i] = get_interval_color_3(idx_color, p)

    Z_layered = Z_colors.reshape(xx.shape[0], xx.shape[1], 3)

    # Plot decision boundaries and data points
    ax.imshow(Z_layered, extent=(x_min, x_max, y_min, y_max),
              origin='lower', vmin=0.0, vmax=1.0)
    ax.scatter(X_2D[:,0], X_2D[:,1], c=y_filt, cmap=cmap_points,
              edgecolors='k', s=15, alpha=0.9)

    # Configure axes
    if show_axes:
        x_ticks = np.linspace(x_min, x_max, 3)
        y_ticks = np.linspace(y_min, y_max, 3)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=16)
        ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgray')
            spine.set_linewidth(0.5)

    # Add accuracy score
    ax.text(x_max - 0.2, y_min + 0.2,
            f"{score_val:.2f}", 
            size=20,
            ha="right", va="bottom",
            bbox=dict(facecolor="white", 
                     alpha=0.8,
                     boxstyle="round,pad=0.3",
                     edgecolor='black',
                     linewidth=0.8))

#------------------------------------------------------------------------------
# Analysis Functions
#------------------------------------------------------------------------------
def calculate_silhouette_scores(X_784, y, pca):
    # Calculate clustering quality in 2D space
    X_filt, y_filt = filter_label_234(X_784, y)
    X_2D = pca.transform(X_filt)
    try:
        sil_score = silhouette_score(X_2D, y_filt)
    except:
        sil_score = float('nan')
    return sil_score

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load and prepare data
    print("\nLoading model data...")
    sub_bp, y_bp, W_bp, b_bp = load_config_data("BP")
    sub_bl, y_bl, W_bl, b_bl = load_config_data("BL")
    sub_blbp, y_blbp, W_blbp, b_blbp = load_config_data("BL->BP")

    # Train classical models
    print("\nTraining classical models...")
    X_bp_clean, _acc_bp_clean = sub_bp["Clean"]
    X_filt, y_filt = filter_label_234(X_bp_clean, y_bp)
    
    clf_svm = SVC(kernel="rbf", gamma="scale", random_state=42, probability=True)
    clf_gp = GaussianProcessClassifier(RBF(1.0), random_state=42, max_iter_predict=500)
    
    clf_svm.fit(X_filt, y_filt)
    clf_gp.fit(X_filt, y_filt)

    # Organize data for visualization
    models_data = {
        "BP": (sub_bp, y_bp, W_bp, b_bp),
        "BL": (sub_bl, y_bl, W_bl, b_bl),
        "BL->BP": (sub_blbp, y_blbp, W_blbp, b_blbp),
        "SVM": (sub_bp, y_bp, None, None),
        "GP": (sub_bp, y_bp, None, None)
    }

    # Prepare PCA transformation
    print("\nFitting PCA...")
    pca_shared = PCA(n_components=2, random_state=42)
    pca_shared.fit(X_bp_clean)
    pca_dict = {method: pca_shared for method in models_data.keys()}

    # Calculate plot boundaries
    print("\nComputing visualization boundaries...")
    all_points = []
    for method_name in pca_dict.keys():
        sub_dict, y_model, _, _ = models_data[method_name]
        for cond in ["Clean", "Noise", "FGSM"]:
            X_cond, _ = sub_dict[cond]
            X_filt_c, _ = filter_label_234(X_cond, y_model)
            X_2D_cond = pca_shared.transform(X_filt_c)
            all_points.append(X_2D_cond)

    all_points = np.vstack(all_points)
    global_x_min = all_points[:,0].min() - 1
    global_x_max = all_points[:,0].max() + 1
    global_y_min = all_points[:,1].min() - 1
    global_y_max = all_points[:,1].max() + 1

    method_ranges = {
        method: (global_x_min, global_x_max, global_y_min, global_y_max) 
        for method in pca_dict.keys()
    }

    # Set up visualization grid
    print("\nGenerating visualizations...")
    models = ["SVM","GP","BP","BL","BL->BP"]
    conditions = ["Clean","Noise","FGSM"]
    row_titles = ["Clean","Noise","FGSM"]
    col_titles = ["SVM-RBF","GP-RBF","BP","BL","BL->BP"]

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(19, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = axes.ravel()

    # Generate subplots
    for idx_subplot, (row_idx, col_idx) in enumerate([(i,j) for i in range(3) for j in range(5)]):
        ax = axes[idx_subplot]
        cond = conditions[row_idx]
        model_name = models[col_idx]

        # Add titles
        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=30)
        if col_idx == 0:
            ax.set_ylabel(row_titles[row_idx], fontsize=30)

        # Get data for current subplot
        sub_dict, y_model, W_model, b_model = models_data[model_name]
        X_cond, best_acc_cond = sub_dict[cond]
        pca_model = pca_dict[model_name]
        x_min_val, x_max_val, y_min_val, y_max_val = method_ranges[model_name]

        # Create visualization
        show_axes = (col_idx == 0)
        if model_name in ("SVM","GP"):
            clf_est = clf_svm if model_name=="SVM" else clf_gp
            plot_sub_manual_pca_layered(
                ax, X_cond, y_model, pca_model,
                x_min_val, x_max_val, y_min_val, y_max_val,
                clf_sklearn=clf_est,
                show_axes=show_axes
            )
        else:
            plot_sub_manual_pca_layered(
                ax, X_cond, y_model, pca_model,
                x_min_val, x_max_val, y_min_val, y_max_val,
                W=W_model, b=b_model, best_acc=best_acc_cond,
                show_axes=show_axes
            )

    plt.tight_layout()

    # Add legends and annotations
    print("\nAdding legends...")
    # Color legend
    legend_elements = [
        Patch(facecolor='red', label='Digit 2'),
        Patch(facecolor='green', label='Digit 3'),
        Patch(facecolor='blue', label='Digit 4')
    ]
    fig.legend(handles=legend_elements, 
              loc='lower center', 
              ncol=3, 
              bbox_to_anchor=(0.5, -0.16), 
              fontsize=20,
              markerscale=2,
              handlelength=3,
              handletextpad=0.5)
    
    # Probability legend
    prob_legend_elements = [
        Patch(facecolor=(1.0, 1.0, 1.0),
              label='p < 0.2', 
              edgecolor='black'),
        Patch(facecolor=(1.0, 0.7, 0.7),
              label='0.2 ≤ p < 0.4'),
        Patch(facecolor=(1.0, 0.5, 0.5),
              label='0.4 ≤ p < 0.6'),
        Patch(facecolor=(1.0, 0.3, 0.3),
              label='0.6 ≤ p < 0.8'),
        Patch(facecolor=(1.0, 0.1, 0.1),
              label='p ≥ 0.8')
    ]

    fig.legend(handles=prob_legend_elements, 
              loc='lower center', 
              ncol=5, 
              bbox_to_anchor=(0.5, -0.10), 
              fontsize=20,
              title='      Probability intervals for Red color\n(Same intervals apply to Green and Blue)',
              title_fontsize=22,
              markerscale=2,
              handlelength=3,
              handletextpad=0.5)

    # Save visualizations
    print("\nSaving visualizations...")
    vis_dir = os.path.join(base_path, "visualization", "ANN_2D_visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Main visualization
    out_path = os.path.join(vis_dir, "ANN_2D_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
    print("Saved main visualization to:", out_path)

    # Calculate silhouette scores
    silhouette_scores = {condition: {} for condition in conditions}
    for model_name in models:
        sub_dict, y_model, _, _ = models_data[model_name]
        pca_model = pca_dict[model_name]
        
        for condition in conditions:
            X_cond, _ = sub_dict[condition]
            score = calculate_silhouette_scores(X_cond, y_model, pca_model)
            silhouette_scores[condition][model_name] = score
    
    # Save and display scores
    df_silhouette = pd.DataFrame(silhouette_scores)
    df_silhouette = df_silhouette[conditions]
    
    csv_path = os.path.join(vis_dir, "silhouette_scores.csv")
    df_silhouette.to_csv(csv_path)
    
    print("\nSilhouette Scores:")
    print(df_silhouette.round(3))
    print("\nMean Scores:")
    print("Per Condition:", df_silhouette.mean().round(3))
    print("Per Model:", df_silhouette.mean(axis=1).round(3))

    # Create silhouette heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_silhouette, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Silhouette Scores Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "silhouette_heatmap.png"), dpi=200)
    plt.close()

    #------------------------------------------------------------------------------
    # Zoomed Visualizations
    #------------------------------------------------------------------------------
    # Zoomed-in view of FGSM degradation in BL→BP regime
    fig_zoom, ax_zoom = plt.subplots(figsize=(6, 5))
    X_fgsm_zoom, _ = sub_blbp["FGSM"]

    # Define zoom boundaries
    x_min_zoom, x_max_zoom = -3.0, 1.5
    y_min_zoom, y_max_zoom = -2.0, 2.5

    def plot_sub_manual_pca_layered_zoom(ax, X_784, y, pca, x_min, x_max, y_min, y_max,
                                       W=None, b=None, best_acc=None, show_axes=False):
        X_filt, y_filt = filter_label_234(X_784, y)
        X_2D = pca.transform(X_filt)
        
        # Use finer grid for detailed visualization
        h = 0.05
        xx, yy = np.meshgrid(
            np.arange(x_min - 0.1, x_max + 0.1, h),
            np.arange(y_min - 0.1, y_max + 0.1, h)
        )
        grid_2D = np.c_[xx.ravel(), yy.ravel()]
        grid_784 = pca.inverse_transform(grid_2D)

        # Calculate probabilities
        Z_probs = predict_custom_784_proba_234(grid_784, W, b)
        classes_ = np.array([2,3,4])

        Z_colors = np.zeros((len(grid_2D), 3), dtype=np.float32)
        for i in range(len(grid_2D)):
            row_probs = Z_probs[i]
            c_ = np.argmax(row_probs)
            p = row_probs[c_]
            label_win = classes_[c_]
            idx_color = LABEL_TO_IDX[label_win]
            Z_colors[i] = get_interval_color_3(idx_color, p)

        Z_layered = Z_colors.reshape(xx.shape[0], xx.shape[1], 3)

        # Plot visualization
        ax.imshow(Z_layered, extent=(x_min, x_max, y_min, y_max),
                  origin='lower', vmin=0.0, vmax=1.0, aspect='auto')
        ax.scatter(X_2D[:,0], X_2D[:,1], c=y_filt, cmap=cmap_points,
                  edgecolors='k', s=15, alpha=0.9)

        if show_axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(np.linspace(x_min, x_max, 5))
            ax.set_yticks(np.linspace(y_min, y_max, 5))
            ax.tick_params(axis='both', which='major', labelsize=12)

        ax.text(x_max - 0.1, y_min + 0.1, f"{best_acc:.2f}", 
                size=14, ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.6, boxstyle="round", pad=0.4))

    # Create zoomed plots for each method
    # BL→BP FGSM
    plot_sub_manual_pca_layered_zoom( 
        ax_zoom,
        X_fgsm_zoom,
        y_blbp,
        pca_shared,
        x_min_zoom,
        x_max_zoom,
        y_min_zoom,
        y_max_zoom,
        W=W_blbp,
        b=b_blbp,
        best_acc=sub_blbp["FGSM"][1],
        show_axes=True
    )

    ax_zoom.set_title("BL→BP FGSM Attack (Zoomed)", fontsize=14)
    zoom_path = os.path.join(vis_dir, "pca_zoom_blbp_fgsm.png")
    plt.savefig(zoom_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    print("Saved BL→BP zoomed FGSM plot to:", zoom_path)
    plt.close(fig_zoom)

    # BP FGSM
    fig_zoom_bp, ax_zoom_bp = plt.subplots(figsize=(6, 5))
    X_fgsm_zoom_bp, _ = sub_bp["FGSM"]

    plot_sub_manual_pca_layered_zoom(
        ax_zoom_bp,
        X_fgsm_zoom_bp,
        y_bp,
        pca_shared,
        x_min_zoom,
        x_max_zoom,
        y_min_zoom,
        y_max_zoom,
        W=W_bp,
        b=b_bp,
        best_acc=sub_bp["FGSM"][1],
        show_axes=True
    )

    ax_zoom_bp.set_title("BP FGSM Attack (Zoomed)", fontsize=14)
    zoom_path_bp = os.path.join(vis_dir, "pca_zoom_bp_fgsm.png")
    plt.savefig(zoom_path_bp, dpi=150, bbox_inches='tight', pad_inches=0.2)
    print("Saved BP zoomed FGSM plot to:", zoom_path_bp)
    plt.close(fig_zoom_bp)

    # BL FGSM
    fig_zoom_bl, ax_zoom_bl = plt.subplots(figsize=(6, 5))
    X_fgsm_zoom_bl, _ = sub_bl["FGSM"]

    plot_sub_manual_pca_layered_zoom(
        ax_zoom_bl,
        X_fgsm_zoom_bl,
        y_bl,
        pca_shared,
        x_min_zoom,
        x_max_zoom,
        y_min_zoom,
        y_max_zoom,
        W=W_bl,
        b=b_bl,
        best_acc=sub_bl["FGSM"][1],
        show_axes=True
    )

    ax_zoom_bl.set_title("BL FGSM Attack (Zoomed)", fontsize=14)
    zoom_path_bl = os.path.join(vis_dir, "pca_zoom_bl_fgsm.png")
    plt.savefig(zoom_path_bl, dpi=150, bbox_inches='tight', pad_inches=0.2)
    print("Saved BL zoomed FGSM plot to:", zoom_path_bl)
    plt.close(fig_zoom_bl)