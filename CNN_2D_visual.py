#!/usr/bin/env python3
"""
CNN_2D_visual.py

Visualizes CNN model performance on MNIST digits 2,3,4 in a 3×5 grid:
- Rows: Clean, Noise, FGSM data
- Columns: SVM, GP, BP, BL, BL->BP methods
- Each neural network uses its own PCA fit on clean data
- SVM/GP share BP's PCA transformation
- Shows decision boundaries and confidence through color intensity
"""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import shuffle
import tensorflow as tf

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
LABEL_TO_IDX = {2: 0, 3: 1, 4: 2}
BASE_COLORS_3 = [
    (1.0, 0.0, 0.0),   # digit=2 => red
    (0.0, 1.0, 0.0),   # digit=3 => green
    (0.0, 0.0, 1.0)    # digit=4 => blue
]
cmap_points = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

configs = {
    "BP": {
        "folder_csv": "cnn_mnist_nobias_backprop",
        "weights_folder": "cnn_nobias_backprop"
    },
    "BL": {
        "folder_csv": "cnn_mnist_nobias_biprop",
        "weights_folder": "cnn_nobias_biprop"
    },
    "BL->BP": {
        "folder_csv": "cnn_mnist_nobias_halfbiprop",
        "weights_folder": "cnn_nobias_halfbiprop"
    }
}

#------------------------------------------------------------------------------
# Data Loading and Processing Functions
#------------------------------------------------------------------------------
def parse_summary_best_block(summary_txt_path):
    """Extracts best accuracies for Clean/Noise/FGSM from summary file."""
    best_clean = best_noise = best_fgsm = 0.0
    
    if not os.path.exists(summary_txt_path):
        return best_clean, best_noise, best_fgsm
    
    with open(summary_txt_path, "r") as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip().lower()
        if "best accuracy test data:" in line:
            j = i + 1
            while j < len(lines):
                l2 = lines[j].strip().lower()
                if "test data:" in l2 and "noise" not in l2 and "fgsm" not in l2:
                    val_str = l2.split(":")[-1].strip()
                    if val_str:
                        best_clean = float(val_str)
                elif "test data with noise:" in l2:
                    val_str = l2.split(":")[-1].strip()
                    if val_str:
                        best_noise = float(val_str)
                elif "test data with fgsm:" in l2:
                    val_str = l2.split(":")[-1].strip()
                    if val_str:
                        best_fgsm = float(val_str)
                if "best accuracy" in l2 and not "test data:" in l2:
                    break
                j += 1
            break
        i += 1
    return best_clean, best_noise, best_fgsm

def filter_label_234(X, y):
    """Filters data to only include samples with labels 2, 3, or 4."""
    if np.all(np.isin(y, [2, 3, 4])):
        return X, y
    mask = np.isin(y, [2, 3, 4])
    return X[mask], y[mask]

def ensure_correct_shape(X):
    """Ensures input array has shape (N, 784) for PCA processing."""
    if X.ndim == 2 and X.shape[1] == 784:
        return X
    elif X.ndim == 4 and X.shape[1:] == (28,28,1):
        return X.reshape(X.shape[0], -1)
    elif X.ndim == 3 and X.shape[1:] == (28,28):
        return X.reshape(X.shape[0], -1)
    else:
        raise ValueError(f"Unexpected shape {X.shape}")

#------------------------------------------------------------------------------
# Model Creation and Loading Functions
#------------------------------------------------------------------------------
def create_cnn_model_nobias():
    """Creates CNN with 2 conv + 2 dense layers, no biases."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (4,4), strides=2, padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2D(128, (4,4), strides=2, padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, use_bias=False),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(10, use_bias=False)
    ])
    return model

def load_cnn_weights_nobias(model, weights_dir):
    """Loads pre-trained weights into CNN model."""
    weight_files = ["C_W1.npy", "C_W2.npy", "C_W3.npy", "C_W4.npy"]
    layer_indices = [0, 2, 5, 7]
    for fname, idx in zip(weight_files, layer_indices):
        w = np.load(os.path.join(weights_dir, fname))
        model.layers[idx].set_weights([w])

def load_config_data(method_name):
    """Loads data and model for a specific method (BP, BL, BL->BP)."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    folder_csv = configs[method_name]["folder_csv"]
    weights_folder = configs[method_name]["weights_folder"]
    
    summary_path = os.path.join(base_path, "csv", folder_csv, "cnn_summary.txt")
    best_clean, best_noise, best_fgsm = parse_summary_best_block(summary_path)
    
    path_clean = os.path.join(base_path, "csv", folder_csv, "cnn_test_subset_234_images.npy")
    path_noise = os.path.join(base_path, "csv", folder_csv, "cnn_test_subset_234_noise.npy")
    path_fgsm = os.path.join(base_path, "csv", folder_csv, "cnn_test_subset_234_fgsm.npy")
    path_labels = os.path.join(base_path, "csv", folder_csv, "cnn_test_subset_234_labels.npy")
    
    for p in [path_clean, path_noise, path_fgsm, path_labels]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
    
    X_clean = ensure_correct_shape(np.load(path_clean))
    X_noise = ensure_correct_shape(np.load(path_noise))
    X_fgsm = ensure_correct_shape(np.load(path_fgsm))
    y_234 = np.load(path_labels)
    
    model = create_cnn_model_nobias()
    weights_dir = os.path.join(base_path, "Weights", weights_folder)
    
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Missing weights directory: {weights_dir}")
    
    load_cnn_weights_nobias(model, weights_dir)
    
    subsets = {
        "Clean": (X_clean, best_clean),
        "Noise": (X_noise, best_noise),
        "FGSM": (X_fgsm, best_fgsm)
    }
    return subsets, y_234, model

#------------------------------------------------------------------------------
# Prediction and Visualization Functions
#------------------------------------------------------------------------------
def predict_cnn_proba_234(X_784, cnn_model):
    """Gets class probabilities for digits 2,3,4 from CNN model."""
    if X_784.ndim == 2 and X_784.shape[1] == 784:
        images = X_784.reshape(-1,28,28,1)
    else:
        images = X_784
    
    logits = cnn_model(images, training=False).numpy()
    shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shift)
    sums_10 = exp_vals.sum(axis=1, keepdims=True)
    softmax_10 = exp_vals / sums_10
    
    partial = softmax_10[:, [2,3,4]]
    sums_234 = partial.sum(axis=1, keepdims=True)
    sums_234[sums_234 == 0.] = 1e-9
    return partial / sums_234

def get_interval_color_3(class_idx, p):
    """Gets color based on class and probability interval."""
    if p < 0.2:
        return (1.0,1.0,1.0)
    
    base_r, base_g, base_b = BASE_COLORS_3[class_idx]
    factor = 0.9 if p >= 0.8 else 0.7 if p >= 0.6 else 0.5 if p >= 0.4 else 0.3
    
    r = (1.0 - factor) + factor * base_r
    g = (1.0 - factor) + factor * base_g
    b = (1.0 - factor) + factor * base_b
    return (r, g, b)

def plot_sub_manual_pca_layered(ax, X_784, y_234, pca, x_min, x_max, y_min, y_max,
                               method_name, clf_sklearn=None, cnn_model=None, 
                               best_acc=None, show_axes=False):
    """Creates layered PCA visualization for a single subplot."""
    X_filt, y_filt = filter_label_234(X_784, y_234)
    X_2D = pca.transform(X_filt)
    
    h = 0.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid_2D = np.c_[xx.ravel(), yy.ravel()]
    grid_784 = pca.inverse_transform(grid_2D)
    
    if clf_sklearn is not None:
        Z_probs = clf_sklearn.predict_proba(grid_784)
        classes_ = clf_sklearn.classes_
        score_val = clf_sklearn.score(X_filt, y_filt)
    else:
        Z_probs = predict_cnn_proba_234(grid_784, cnn_model)
        classes_ = np.array([2,3,4])
        score_val = best_acc if best_acc is not None else 0.0
    
    n_points = len(grid_2D)
    Z_colors = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
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
    
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgray')
            spine.set_linewidth(0.5)
    
    ax.text(
        x_max, y_min, f"{score_val:.2f}",
        size=20,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.6, boxstyle="round", pad=0.4)
    )

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))

    print("\nLoading data and models...")
    sub_bp, y_bp, model_bp = load_config_data("BP")
    sub_bl, y_bl, model_bl = load_config_data("BL")
    sub_blbp, y_blbp, model_blbp = load_config_data("BL->BP")

    print("\nTraining classical models...")
    X_bp_clean, best_acc_bp_clean = sub_bp["Clean"]
    X_filt_bp, y_filt_bp = filter_label_234(X_bp_clean, y_bp)
    X_filt_bp, y_filt_bp = shuffle(X_filt_bp, y_filt_bp, random_state=42)
    
    clf_svm = SVC(kernel="rbf", gamma="scale", probability=True, random_state=42)
    clf_gp = GaussianProcessClassifier(RBF(1.0), random_state=42, max_iter_predict=500)
    
    clf_svm.fit(X_filt_bp, y_filt_bp)
    clf_gp.fit(X_filt_bp, y_filt_bp)

    data_models = {
        "SVM": (sub_bp, y_bp, clf_svm, None),
        "GP": (sub_bp, y_bp, clf_gp, None),
        "BP": (sub_bp, y_bp, model_bp, None),
        "BL": (sub_bl, y_bl, model_bl, None),
        "BL->BP": (sub_blbp, y_blbp, model_blbp, None)
    }

    print("\nFitting PCA transformations...")
    pca_dict = {}
    for m_name in ["BP", "BL", "BL->BP"]:
        X_clean_m, _ = data_models[m_name][0]["Clean"]
        pca_m = PCA(n_components=2, random_state=42)
        pca_m.fit(X_clean_m)
        pca_dict[m_name] = pca_m
    
    pca_dict["SVM"] = pca_dict["BP"]
    pca_dict["GP"] = pca_dict["BP"]

    print("\nCalculating visualization boundaries...")
    method_ranges = {}
    for m_name in ["BP", "BL", "BL->BP", "SVM", "GP"]:
        sub_dict, y_sub, predictor, _ = data_models[m_name]
        X_local_clean, _ = sub_dict["Clean"]
        X_local_2D = pca_dict[m_name].transform(X_local_clean)
        
        x_min = X_local_2D[:,0].min() - 1
        x_max = X_local_2D[:,0].max() + 1
        y_min = X_local_2D[:,1].min() - 1
        y_max = X_local_2D[:,1].max() + 1
        method_ranges[m_name] = (x_min, x_max, y_min, y_max)

    print("\nGenerating visualization grid...")
    row_titles = ["Clean", "Noise", "FGSM"]
    col_titles = ["SVM", "GP", "BP", "BL", "BL->BP"]
    rows = ["Clean", "Noise", "FGSM"]
    cols = ["SVM", "GP", "BP", "BL", "BL->BP"]

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(19,12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = axes.ravel()

    print("\nGenerating individual subplots...")
    for i, (r_idx, c_idx) in enumerate([(r,c) for r in range(3) for c in range(5)]):
        ax = axes[i]
        cond_name = rows[r_idx]
        method_name = cols[c_idx]
        
        if r_idx == 0:
            ax.set_title(col_titles[c_idx], fontsize=30)
        if c_idx == 0:
            ax.set_ylabel(row_titles[r_idx], fontsize=30)
        
        sub_dict, y_sub, predictor, _ = data_models[method_name]
        X_cond, best_acc_cond = sub_dict[cond_name]
        pca_m = pca_dict[method_name]
        x_min_val, x_max_val, y_min_val, y_max_val = method_ranges[method_name]

        if method_name in ["SVM", "GP"]:
            plot_sub_manual_pca_layered(
                ax, X_cond, y_sub, pca_m,
                x_min_val, x_max_val, y_min_val, y_max_val,
                method_name=method_name,
                clf_sklearn=predictor,
                cnn_model=None,
                best_acc=None,
                show_axes=False
            )
        else:
            plot_sub_manual_pca_layered(
                ax, X_cond, y_sub, pca_m,
                x_min_val, x_max_val, y_min_val, y_max_val,
                method_name=method_name,
                clf_sklearn=None,
                cnn_model=predictor,
                best_acc=best_acc_cond,
                show_axes=False
            )

    print("\nAdding legends...")
    legend_elements = [
        Patch(facecolor='red', label='Digit 2'),
        Patch(facecolor='green', label='Digit 3'),
        Patch(facecolor='blue', label='Digit 4')
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, -0.16),
        fontsize=20,
        markerscale=2,
        handlelength=3,
        handletextpad=0.5
    )

    prob_legend_elements = [
        Patch(facecolor=(1.0, 1.0, 1.0), label='p < 0.2', edgecolor='black'),
        Patch(facecolor=(1.0, 0.7, 0.7), label='0.2 ≤ p < 0.4'),
        Patch(facecolor=(1.0, 0.5, 0.5), label='0.4 ≤ p < 0.6'),
        Patch(facecolor=(1.0, 0.3, 0.3), label='0.6 ≤ p < 0.8'),
        Patch(facecolor=(1.0, 0.1, 0.1), label='p ≥ 0.8')
    ]
    fig.legend(
        handles=prob_legend_elements,
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0.5, -0.10),
        fontsize=20,
        title='Probability intervals for Red color\n(Same intervals apply to Green and Blue)',
        title_fontsize=22,
        markerscale=2,
        handlelength=3,
        handletextpad=0.5
    )

    plt.tight_layout()
    
    print("\nSaving visualization...")
    os.makedirs(os.path.join("visualization", "CNN_2D_visualization"), exist_ok=True)
    out_path = os.path.join("visualization", "CNN_2D_visualization", "cnn_visualization_sharedPCA1.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
    print(f"Saved figure to: {out_path}")

    print("\n=== Explanation of PCA usage ===")
    print("In this script, each method (BP, BL, BL->BP) gets its own PCA fit on that method's Clean subset.")
    print("SVM/GP re-use BP's PCA, because they also train on BP data.")
    print("We do NOT re-fit PCA for Noise/FGSM => the background is consistent, just the data points move.")