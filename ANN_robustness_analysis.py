"""
ANN_robustness_analysis.py

Performs comprehensive robustness analysis of neural network models (BP, BL, BL->BP)
comparing their performance on clean, noisy, and FGSM-perturbed data. Includes:
- PCA visualization for each method
- Silhouette score analysis
- Performance comparison plots
- Robustness metrics calculation
"""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
print(f"Using local path: {base_path}")

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

# Verify required directories exist
for config_name, config in configs.items():
    folder_csv = config["folder_csv"]
    folder_weights = config["folder_weights"]
    
    csv_path = os.path.join(base_path, "csv", folder_csv)
    weights_path = os.path.join(base_path, "Weights", folder_weights)
    
    print(f"Checking paths for {config_name}:")
    print(f"  CSV path: {csv_path} - {'EXISTS' if os.path.exists(csv_path) else 'MISSING'}")
    print(f"  Weights path: {weights_path} - {'EXISTS' if os.path.exists(weights_path) else 'MISSING'}")

#------------------------------------------------------------------------------
# Visualization Constants
#------------------------------------------------------------------------------
cmap_points = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
LABEL_TO_IDX = {2:0, 3:1, 4:2}
BASE_COLORS_3 = [
    (1.0, 0.0, 0.0),  # label=2 => red
    (0.0, 1.0, 0.0),  # label=3 => green
    (0.0, 0.0, 1.0)   # label=4 => blue
]

#------------------------------------------------------------------------------
# Data Loading Functions
#------------------------------------------------------------------------------
def parse_summary_best_block(summary_txt_path):
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
    print(f"Loading {config_name} data...")
    
    folder_csv = configs[config_name]["folder_csv"]
    folder_w   = configs[config_name]["folder_weights"]

    summary_path = os.path.join(base_path, "csv", folder_csv, "ANN_summary.txt")
    best_clean, best_noise, best_fgsm = parse_summary_best_block(summary_path)

    path_clean  = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_images.npy")
    path_noise  = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_noise.npy")
    path_fgsm   = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_fgsm.npy")
    path_labels = os.path.join(base_path, "csv", folder_csv, "ANN_test_subset_234_labels.npy")

    for path, name in [(path_clean, "clean images"), (path_noise, "noisy images"), 
                       (path_fgsm, "FGSM images"), (path_labels, "labels")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found at {path}")
            sys.exit(1)

    X_clean = np.load(path_clean)
    X_noise = np.load(path_noise)
    X_fgsm  = np.load(path_fgsm)
    y_234   = np.load(path_labels)

    w_path = os.path.join(base_path, "Weights", folder_w, "ANN_C_W1.npy")
    b_path = os.path.join(base_path, "Weights", folder_w, "ANN_C_B1.npy")
    
    W = np.load(w_path)
    if os.path.exists(b_path):
        b = np.load(b_path)
    else:
        print(f"Using zero bias for {config_name} (nobias model)")
        b = np.zeros(W.shape[1])

    print("DEBUG:", config_name, "W.shape =", W.shape, "b.shape =", b.shape)

    subsets = {
        "Clean": (X_clean, best_clean),
        "Noise": (X_noise, best_noise),
        "FGSM":  (X_fgsm,  best_fgsm)
    }
    
    return subsets, y_234, W, b

#------------------------------------------------------------------------------
# Model Prediction Functions
#------------------------------------------------------------------------------
def predict_custom_784_proba_234(X_784, W, b):
    logits_10 = X_784 @ W + b
    shift = logits_10 - np.max(logits_10, axis=1, keepdims=True)
    exp_vals = np.exp(shift)
    sums_10 = np.sum(exp_vals, axis=1, keepdims=True)
    probs_10 = exp_vals / sums_10

    partial = probs_10[:, [2,3,4]]
    sums_234 = np.sum(partial, axis=1, keepdims=True)
    probs_3 = partial / sums_234
    return probs_3

def filter_label_234(X, y):
    mask = np.isin(y, [2,3,4])
    return X[mask], y[mask]

#------------------------------------------------------------------------------
# Visualization Helper Functions
#------------------------------------------------------------------------------
def get_interval_color_3(class_idx, p):
    if p < 0.2:
        return (1.0,1.0,1.0)

    base_r, base_g, base_b = BASE_COLORS_3[class_idx]
    if p < 0.4:
        factor = 0.3
    elif p < 0.6:
        factor = 0.5
    elif p < 0.8:
        factor = 0.7
    else:
        factor = 0.9

    r = (1.0 - factor) + factor * base_r
    g = (1.0 - factor) + factor * base_g
    b = (1.0 - factor) + factor * base_b
    return (r, g, b)

def plot_sub_manual_pca_layered(ax, X_784, y, pca, x_min, x_max, y_min, y_max,
                               clf_sklearn=None, W=None, b=None, best_acc=None, 
                               show_axes=False):
    X_filt, y_filt = filter_label_234(X_784, y)
    X_2D = pca.transform(X_filt)

    try:
        sil_score = silhouette_score(X_2D, y_filt)
    except:
        sil_score = float('nan')
    
    h = 0.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid_2D = np.c_[xx.ravel(), yy.ravel()]
    grid_784 = pca.inverse_transform(grid_2D)

    if clf_sklearn is not None:
        Z_probs = clf_sklearn.predict_proba(grid_784)
        score_val = clf_sklearn.score(X_filt, y_filt)
        classes_ = clf_sklearn.classes_
    else:
        Z_probs = predict_custom_784_proba_234(grid_784, W, b)
        score_val = best_acc
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

    ax.imshow(Z_layered, extent=(x_min, x_max, y_min, y_max),
              origin='lower', vmin=0.0, vmax=1.0)
    ax.scatter(X_2D[:,0], X_2D[:,1], c=y_filt, cmap=cmap_points,
              edgecolors='k', s=15, alpha=0.9)

    if show_axes:
        x_ticks = np.linspace(x_min, x_max, 3)
        y_ticks = np.linspace(y_min, y_max, 3)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=24)
        ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgray')
            spine.set_linewidth(0.5)

    ax.text(x_max, y_min, 
            f"Acc: {score_val:.2f}\nSil: {sil_score:.2f}", 
            size=24,
            ha="right", va="bottom",
            bbox=dict(facecolor="white", 
                     alpha=0.8,
                     boxstyle="round",
                     pad=0.6,
                     edgecolor='black'),
            weight='bold')

#------------------------------------------------------------------------------
# Analysis Functions
#------------------------------------------------------------------------------
def calculate_silhouette_scores(X_784, y, pca):
    X_filt, y_filt = filter_label_234(X_784, y)
    X_2D = pca.transform(X_filt)
    try:
        sil_score = silhouette_score(X_2D, y_filt)
    except:
        sil_score = float('nan')
    return sil_score

#------------------------------------------------------------------------------
# Performance Analysis Functions
#------------------------------------------------------------------------------
def create_performance_drops_plots(accuracy_data, vis_dir):
    SMALL_SIZE = 24
    MEDIUM_SIZE = 28
    LARGE_SIZE = 32
    BIGGER_SIZE = 36
    VALUE_SIZE = 24

    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=LARGE_SIZE)
    plt.rc('axes', labelsize=LARGE_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    
    models = ["BP", "BL", "BL->BP"]
    clean_acc = [accuracy_data[model]["Clean"] for model in models]
    noise_acc = [accuracy_data[model]["Noise"] for model in models]
    fgsm_acc = [accuracy_data[model]["FGSM"] for model in models]
    
    plt.figure(figsize=(14, 10))
    ax1 = plt.gca()
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, clean_acc, width, label='Clean', color='lightgreen', edgecolor='darkgreen')
    bars2 = ax1.bar(x, noise_acc, width, label='Noise', color='lightblue', edgecolor='darkblue')
    bars3 = ax1.bar(x + width, fgsm_acc, width, label='FGSM', color='salmon', edgecolor='darkred')
    
    ax1.set_ylabel('Accuracy (0-1 scale)', fontsize=LARGE_SIZE)
    ax1.set_title('Raw Model Performance\n(higher is better)', fontsize=BIGGER_SIZE, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, fontsize=LARGE_SIZE)
    ax1.tick_params(axis='both', labelsize=LARGE_SIZE)
    
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=VALUE_SIZE)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=LARGE_SIZE,
              bbox_to_anchor=(0.5, -0.15),
              loc='upper center',
              ncol=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'performance_raw.png'), 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.3)
    plt.close()
    
    plt.figure(figsize=(14, 10))
    ax2 = plt.gca()
    
    clean_baseline = np.array(clean_acc)
    noise_drop = (clean_baseline - noise_acc) / clean_baseline * 100
    fgsm_drop = (clean_baseline - fgsm_acc) / clean_baseline * 100
    
    bars4 = ax2.bar(x - width/2, noise_drop, width, label='Drop due to Noise', 
                    color='lightblue', edgecolor='darkblue')
    bars5 = ax2.bar(x + width/2, fgsm_drop, width, label='Drop due to FGSM', 
                    color='salmon', edgecolor='darkred')
    
    ax2.set_ylabel('Performance Drop (%)\n(lower is better)', fontsize=LARGE_SIZE)
    ax2.set_title('Performance Degradation\nunder Attacks', fontsize=BIGGER_SIZE, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, fontsize=LARGE_SIZE)
    ax2.tick_params(axis='both', labelsize=LARGE_SIZE)
    
    def add_value_labels_percent(bars):
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=VALUE_SIZE)
    
    add_value_labels_percent(bars4)
    add_value_labels_percent(bars5)
    
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=LARGE_SIZE,
              bbox_to_anchor=(0.5, -0.15),
              loc='upper center',
              ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'performance_drops.png'), 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.3)
    plt.close()

def create_robustness_analysis_plot(accuracy_data, vis_dir):
    SMALL_SIZE = 24
    MEDIUM_SIZE = 28
    LARGE_SIZE = 32
    BIGGER_SIZE = 36
    VALUE_SIZE = 24

    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=LARGE_SIZE)
    plt.rc('axes', labelsize=LARGE_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    
    models = ["BP", "BL", "BL->BP"]
    robustness_scores = {}
    
    for model in models:
        clean_baseline = accuracy_data[model]["Clean"]
        noise_resilience = accuracy_data[model]["Noise"] / clean_baseline
        fgsm_resilience = accuracy_data[model]["FGSM"] / clean_baseline
        avg_resilience = (noise_resilience + fgsm_resilience) / 2
        
        robustness_scores[model] = {
            'Noise Resilience': noise_resilience,
            'FGSM Resilience': fgsm_resilience,
            'Average Resilience': avg_resilience
        }
    
    df_robustness = pd.DataFrame(robustness_scores).T
    
    plt.figure(figsize=(14, 10))
    ax1 = plt.gca()
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, df_robustness['Noise Resilience'], width,
                    label='Noise Resilience', color='lightblue', edgecolor='darkblue')
    bars2 = ax1.bar(x, df_robustness['FGSM Resilience'], width,
                    label='FGSM Resilience', color='salmon', edgecolor='darkred')
    bars3 = ax1.bar(x + width, df_robustness['Average Resilience'], width,
                    label='Average Resilience', color='lightgreen', edgecolor='darkgreen')
    
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=VALUE_SIZE)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax1.set_title('Model Resilience Scores\n(closer to 1.0 is better)', pad=20)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Resilience Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.text(-0.5, 1.01, 'Perfect Resilience (1.0)', fontsize=MEDIUM_SIZE, alpha=0.5)
    
    ax1.legend(bbox_to_anchor=(0.5, -0.15),
              loc='upper center',
              ncol=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'resilience_scores.png'), 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.3)
    plt.close()
    
    plt.figure(figsize=(14, 10))
    ax2 = plt.gca()
    
    overall_scores = df_robustness['Average Resilience'].sort_values(ascending=False)
    colors = ['gold', 'silver', '#cd7f32']
    
    bars = ax2.bar(range(len(models)), overall_scores, 
                   color=colors, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=VALUE_SIZE)
    
    ax2.set_title('Overall Model Ranking\nBased on Average Resilience', pad=20)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Average Resilience Score')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(overall_scores.index, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'model_ranking.png'), 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.3)
    plt.close()
    
    return df_robustness

#------------------------------------------------------------------------------
# Architecture Comparison Functions
#------------------------------------------------------------------------------
def create_ann_cnn_comparison_plots(base_path, vis_dir):
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    methods = ['backprop', 'biprop', 'halfbiprop']
    architectures = ['ANN', 'CNN']
    
    TITLE_SIZE = 32
    LABEL_SIZE = 28
    TICK_SIZE = 24
    VALUE_SIZE = 24
    LEGEND_SIZE = 24
    
    data = {arch: {method: {} for method in methods} for arch in architectures}
    
    for method in methods:
        summary_path = os.path.join(base_path, "csv", f"mnist_nn_nobias_{method}", "ANN_summary.txt")
        if os.path.exists(summary_path):
            print(f"Found ANN summary for {method}")
            with open(summary_path, 'r') as f:
                lines = f.readlines()
                data['ANN'][method] = {
                    'Clean': float(lines[2].split(': ')[1]),
                    'Noise': float(lines[3].split(': ')[1]),
                    'FGSM': float(lines[4].split(': ')[1])
                }
        else:
            print(f"Missing ANN summary for {method}")
            data['ANN'][method] = {'Clean': 0, 'Noise': 0, 'FGSM': 0}
    
    for method in methods:
        summary_path = os.path.join(base_path, "csv", f"cnn_mnist_nobias_{method}", "cnn_summary.txt")
        if os.path.exists(summary_path):
            print(f"Found CNN summary for {method}")
            with open(summary_path, 'r') as f:
                lines = f.readlines()
                data['CNN'][method] = {
                    'Clean': float(lines[2].split(': ')[1]),
                    'Noise': float(lines[3].split(': ')[1]),
                    'FGSM': float(lines[4].split(': ')[1])
                }
        else:
            print(f"Missing CNN summary for {method}")
            data['CNN'][method] = {'Clean': 0, 'Noise': 0, 'FGSM': 0}
    
    colors = {
        'ANN': {'Clean': '#4878d0', 'Noise': '#ee854a', 'FGSM': '#d65f5f'},
        'CNN': {'Clean': '#82b1ff', 'Noise': '#ffb74d', 'FGSM': '#ff8a80'}
    }
    
    conditions = ['Clean', 'Noise', 'FGSM']
    titles = ['Clean Data Accuracy', 'Noise Robustness', 'FGSM Robustness']
    filenames = ['clean_comparison.png', 'noise_comparison.png', 'fgsm_comparison.png']
    
    for condition, title, filename in zip(conditions, titles, filenames):
        fig = plt.figure(figsize=(12, 9))
        ax = plt.gca()
        
        width = 0.25
        x = np.arange(len(methods))
        
        for i, arch in enumerate(architectures):
            values = [data[arch][method][condition] for method in methods]
            offset = width/2 if i == 1 else -width/2
            bars = ax.bar(x + offset, values, width, 
                         label=f'{arch}',
                         color=colors[arch][condition],
                         edgecolor='black',
                         linewidth=1)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom',
                           fontsize=VALUE_SIZE)
        
        ax.set_ylabel('Accuracy', fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(['BP', 'BL', 'BL->BP'], fontsize=TICK_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        legend = ax.legend(fontsize=LEGEND_SIZE,
                          ncol=2,
                          bbox_to_anchor=(0.5, -0.15),
                          loc='upper center',
                          borderaxespad=0.,
                          frameon=True,
                          edgecolor='black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, filename), 
                    dpi=300, 
                    bbox_inches='tight',
                    pad_inches=0.3)
        plt.close()
        print(f"Saved {condition} comparison plot")

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load data for all methods
    sub_bp, y_bp, W_bp, b_bp = load_config_data("BP")
    sub_bl, y_bl, W_bl, b_bl = load_config_data("BL")
    sub_blbp, y_blbp, W_blbp, b_blbp = load_config_data("BL->BP")

    # Train SVM and GP on BP's clean data
    X_bp_clean, _acc_bp_clean = sub_bp["Clean"]
    X_filt, y_filt = filter_label_234(X_bp_clean, y_bp)
    
    clf_svm = SVC(kernel="rbf", gamma="scale", random_state=42, probability=True)
    clf_gp = GaussianProcessClassifier(RBF(1.0), random_state=42, max_iter_predict=500)
    
    clf_svm.fit(X_filt, y_filt)
    clf_gp.fit(X_filt, y_filt)

    # Collect all data
    models_data = {
        "BP": (sub_bp, y_bp, W_bp, b_bp),
        "BL": (sub_bl, y_bl, W_bl, b_bl),
        "BL->BP": (sub_blbp, y_blbp, W_blbp, b_blbp),
        "SVM": (sub_bp, y_bp, None, None),
        "GP": (sub_bp, y_bp, None, None)
    }

    # Fit shared PCA
    pca_shared = PCA(n_components=2, random_state=42)
    pca_shared.fit(X_bp_clean)
    
    # Use same PCA for all methods
    models = ["BP", "BL", "BL->BP"]
    conditions = ["Clean", "Noise", "FGSM"]
    pca_dict = {method: pca_shared for method in models}
    
    # Create visualization directory
    vis_dir = os.path.join(base_path, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create robustness_analysis subfolder
    robustness_dir = os.path.join(vis_dir, "robustness_analysis")
    os.makedirs(robustness_dir, exist_ok=True)

    # Calculate silhouette scores
    silhouette_scores = {condition: {} for condition in conditions}
    for model_name in models:
        sub_dict, y_model, _, _ = models_data[model_name]
        pca_model = pca_dict[model_name]
        
        for condition in conditions:
            X_cond, _ = sub_dict[condition]
            score = calculate_silhouette_scores(X_cond, y_model, pca_model)
            silhouette_scores[condition][model_name] = score
    
    # Collect accuracy data
    accuracy_data = {}
    for model_name in models:
        sub_dict, _, _, _ = models_data[model_name]
        accuracy_data[model_name] = {
            "Clean": sub_dict["Clean"][1],
            "Noise": sub_dict["Noise"][1],
            "FGSM": sub_dict["FGSM"][1]
        }

    # Create performance analysis plots
    create_performance_drops_plots(accuracy_data, robustness_dir)
    df_robustness = create_robustness_analysis_plot(accuracy_data, robustness_dir)

    # Print analysis results
    print("\n=== Detailed Analysis ===")
    print("\nRobustness Analysis:")
    print(df_robustness.round(3))

    print("\nKey Findings:")
    for model in models:
        clean_acc = accuracy_data[model]["Clean"]
        noise_drop = (clean_acc - accuracy_data[model]["Noise"]) / clean_acc * 100
        fgsm_drop = (clean_acc - accuracy_data[model]["FGSM"]) / clean_acc * 100
        
        print(f"\n{model}:")
        print(f"- Clean accuracy: {clean_acc:.3f}")
        print(f"- Performance drop under noise: {noise_drop:.1f}%")
        print(f"- Performance drop under FGSM: {fgsm_drop:.1f}%")

    base_path = os.getcwd()
    create_ann_cnn_comparison_plots(base_path, robustness_dir)