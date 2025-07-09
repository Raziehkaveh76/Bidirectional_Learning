"""
ANN_animation.py

Creates animated 3D PCA visualizations for comparing different neural network
training methods (BP, BL, BL->BP) and classical models (SVM, GP) on MNIST data.
Generates rotating plots saved as GIF files for clean, noisy, and FGSM-perturbed data.
"""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial import ConvexHull
from scipy.special import softmax
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

#------------------------------------------------------------------------------
# Configuration and Directory Setup
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

def create_animation_directories():
    try:
        vis_dir = os.path.join("visualization_3d")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Created/verified main directory: {vis_dir}")
        
        anim_dir = os.path.join(vis_dir, "animations")
        os.makedirs(anim_dir, exist_ok=True)
        print(f"Created/verified animation directory: {anim_dir}")
        
        temp_dir = os.path.join(anim_dir, "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Created/verified temporary frames directory: {temp_dir}")
        
        return anim_dir
        
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        sys.exit(1)

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
cmap_points = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
LABEL_TO_IDX = {2:0, 3:1, 4:2}
BASE_COLORS_3 = [
    (1.0, 0.0, 0.0),  # label=2 => red
    (0.0, 1.0, 0.0),  # label=3 => green
    (0.0, 0.0, 1.0)   # label=4 => blue
]

#------------------------------------------------------------------------------
# Data Loading and Processing Functions
#------------------------------------------------------------------------------
def parse_summary_best_block(summary_txt_path):
    best_clean = 0.0
    best_noise = 0.0
    best_fgsm  = 0.0

    if not os.path.exists(summary_txt_path):
        print(f"WARNING: Summary file not found at {summary_txt_path}")
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
    folder_w = configs[config_name]["folder_weights"]

    summary_path = os.path.join("csv", folder_csv, "ANN_summary.txt")
    path_clean = os.path.join("csv", folder_csv, "ANN_test_subset_234_images.npy")
    path_noise = os.path.join("csv", folder_csv, "ANN_test_subset_234_noise.npy")
    path_fgsm = os.path.join("csv", folder_csv, "ANN_test_subset_234_fgsm.npy")
    path_labels = os.path.join("csv", folder_csv, "ANN_test_subset_234_labels.npy")

    best_clean, best_noise, best_fgsm = parse_summary_best_block(summary_path)

    for path, name in [(path_clean, "clean images"), (path_noise, "noisy images"), 
                       (path_fgsm, "FGSM images"), (path_labels, "labels")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found at {path}")
            sys.exit(1)

    X_clean = np.load(path_clean)
    X_noise = np.load(path_noise)
    X_fgsm = np.load(path_fgsm)
    y_234 = np.load(path_labels)

    w_path = os.path.join(base_path, "Weights", folder_w, "ANN_C_W1.npy")
    
    if not os.path.exists(w_path):
        print(f"ERROR: Weight file not found at {w_path}")
        sys.exit(1)
        
    W = np.load(w_path)
    b = np.zeros(W.shape[1])

    print(f"Loaded {config_name} data: W.shape={W.shape}, b.shape={b.shape}, X_clean.shape={X_clean.shape}")

    return {
        "Clean": (X_clean, best_clean),
        "Noise": (X_noise, best_noise),
        "FGSM": (X_fgsm, best_fgsm)
    }, y_234, W, b

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
# Model Training and PCA Setup
#------------------------------------------------------------------------------
print("Loading BP data for SVM/GP training...")
sub_bp, y_bp, W_bp, b_bp = load_config_data("BP")
X_bp_clean, _acc_bp_clean = sub_bp["Clean"]
X_filt, y_filt = filter_label_234(X_bp_clean, y_bp)

print(f"Training SVM and GP on filtered data: {X_filt.shape}...")
clf_svm = SVC(kernel="rbf", gamma="scale", random_state=42, probability=True)
clf_gp  = GaussianProcessClassifier(RBF(1.0), random_state=42, max_iter_predict=500)

clf_svm.fit(X_filt, y_filt)
clf_gp.fit(X_filt, y_filt)

print("SVM classes_:", clf_svm.classes_)
print("GP classes_:",  clf_gp.classes_)

print("Loading BL data...")
sub_bl, y_bl, W_bl, b_bl = load_config_data("BL")
print("Loading BL->BP data...")
sub_blbp, y_blbp, W_blbp, b_blbp = load_config_data("BL->BP")

models_data = {
    "BP":      (sub_bp,   y_bp,   W_bp,   b_bp),
    "BL":      (sub_bl,   y_bl,   W_bl,   b_bl),
    "BL->BP":  (sub_blbp, y_blbp, W_blbp, b_blbp),
    "SVM":     (sub_bp,   y_bp,   None,   None),
    "GP":      (sub_bp,   y_bp,   None,   None)
}

print("Fitting 3D PCAs for each method...")
pca_dict = {}

X_clean_bp, bestacc_bp = models_data["BP"][0]["Clean"]
pca_bp = PCA(n_components=3, random_state=42)
pca_bp.fit(X_clean_bp)
pca_dict["BP"] = pca_bp

X_clean_bl, bestacc_bl = models_data["BL"][0]["Clean"]
pca_bl = PCA(n_components=3, random_state=42)
pca_bl.fit(X_clean_bl)
pca_dict["BL"] = pca_bl

X_clean_blbp, bestacc_blbp = models_data["BL->BP"][0]["Clean"]
pca_blbp = PCA(n_components=3, random_state=42)
pca_blbp.fit(X_clean_blbp)
pca_dict["BL->BP"] = pca_blbp

pca_svmgp = PCA(n_components=3, random_state=42)
pca_svmgp.fit(X_filt)
pca_dict["SVM"] = pca_svmgp
pca_dict["GP"]  = pca_svmgp

#------------------------------------------------------------------------------
# Visualization Functions
#------------------------------------------------------------------------------
def add_convex_hulls(ax, X_3D, y_labels):
    for label in [2, 3, 4]:
        mask = (y_labels == label)
        if np.sum(mask) < 4:
            continue
            
        points = X_3D[mask]
        hull = ConvexHull(points)
        
        color = BASE_COLORS_3[LABEL_TO_IDX[label]]
        
        for simplex in hull.simplices:
            ax.plot3D(points[simplex, 0], points[simplex, 1], 
                     points[simplex, 2], color=color, alpha=0.3)
            
        for simplex in hull.simplices:
            face = Poly3DCollection([points[simplex]])
            face.set_color(color)
            face.set_alpha(0.1)
            ax.add_collection3d(face)

def create_enhanced_3d_plot(
    X_3D, y_labels, 
    title="3D PCA Visualization", 
    accuracy=None,
    save_path=None,
    n_frames=72,
    elevation=30
):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = []
    for label in y_labels:
        idx = LABEL_TO_IDX[label]
        colors.append(BASE_COLORS_3[idx])
    
    scatter = ax.scatter(
        X_3D[:, 0],
        X_3D[:, 1],
        X_3D[:, 2],
        c=colors,
        s=30,
        alpha=0.9,
        edgecolors='k'
    )
    
    add_convex_hulls(ax, X_3D, y_labels)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    
    if accuracy is not None:
        title = f"{title} (Acc: {accuracy:.2f})"
    ax.set_title(title, fontsize=14)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(1,0,0), markersize=10, label='Digit 2'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0,1,0), markersize=10, label='Digit 3'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0,0,1), markersize=10, label='Digit 4')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    if accuracy is not None:
        ax.text2D(0.05, 0.95, f"Accuracy: {accuracy:.2f}", 
                 transform=ax.transAxes,
                 fontsize=14,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    def animate(i):
        ax.view_init(elev=elevation, azim=i * (360 / n_frames))
        return [scatter]
    
    ani = animation.FuncAnimation(
        fig, animate, frames=n_frames, 
        interval=50, blit=True
    )
    
    if save_path:
        temp_dir = os.path.join(os.path.dirname(save_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"Generating {n_frames} frames for animation...")
        for i in range(n_frames):
            ax.view_init(elev=elevation, azim=i * (360 / n_frames))
            plt.savefig(os.path.join(temp_dir, f"frame_{i:03d}.png"), dpi=100)
        
        print(f"Creating GIF animation at {save_path}...")
        with imageio.get_writer(save_path, mode='I', duration=0.05) as writer:
            for i in range(n_frames):
                image = imageio.imread(os.path.join(temp_dir, f"frame_{i:03d}.png"))
                writer.append_data(image)
        
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Animation saved to {save_path}")
    
    return fig, ani

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------
print("Creating animated 3D visualizations...")
models = ["BP", "BL", "BL->BP", "SVM", "GP"]
conditions = ["Clean", "Noise", "FGSM"]

anim_dir = create_animation_directories()

for model_name in models:
    for cond in conditions:
        print(f"Creating enhanced animation for {model_name} - {cond}...")
        
        sub_dict, y_model, W_model, b_model = models_data[model_name]
        X_cond, best_acc_cond = sub_dict[cond]
        pca_model = pca_dict[model_name]
        
        X_filt, y_filt = filter_label_234(X_cond, y_model)
        X_3D = pca_model.transform(X_filt)
        
        if model_name in ("SVM", "GP"):
            clf_est = clf_svm if model_name == "SVM" else clf_gp
            accuracy = clf_est.score(X_filt, y_filt)
        else:
            accuracy = best_acc_cond
            
        save_path = os.path.join(anim_dir, f"{model_name}_{cond}_3D_enhanced.gif")
        fig, ani = create_enhanced_3d_plot(
            X_3D, y_filt,
            title=f"{model_name} - {cond}",
            accuracy=accuracy,
            save_path=save_path
        )
        
        plt.close(fig)

print("All animations created successfully!")