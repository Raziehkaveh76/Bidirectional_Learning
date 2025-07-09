"""
ANN_performance_visual.py

Creates performance visualizations comparing different neural network training methods
(BP, BL, BL->BP) including weight distributions, attack comparisons, and various
performance metrics across clean, noisy, and FGSM-perturbed data.
"""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
# Standard library
import os

# Data processing and analysis
import numpy as np
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#------------------------------------------------------------------------------
# Weight Distribution Visualization
#------------------------------------------------------------------------------
def plot_weights_comparison(base_path, vis_dir):
    methods = ['nobias_backprop', 'nobias_biprop', 'nobias_halfbiprop']
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    figure_width = 7
    figure_height = 9
    fig, axes = plt.subplots(len(methods), 1, figsize=(figure_width, figure_height), sharex=True)
    
    custom_titles = [
        "Backpropagation without bias",
        "BiPropogation without bias",
        "Half BiPropagation without bias"
    ]
    
    for idx, method in enumerate(methods):
        weight_path = os.path.join(base_path, "Weights", method, "ANN_C_W1.npy")
        if os.path.exists(weight_path):
            weights = np.load(weight_path)
            
            n, bins, patches = axes[idx].hist(weights.flatten(), bins=25,
                                             alpha=0.8, color='#4878d0', 
                                             edgecolor='black', linewidth=0.5,
                                             label='Weight Distribution')
            
            density = stats.gaussian_kde(weights.flatten())
            x_vals = np.linspace(weights.min(), weights.max(), 60)
            axes[idx].plot(x_vals, density(x_vals) * len(weights.flatten()) * (bins[1] - bins[0]), 
                          color='#d65f5f', linewidth=1.0,
                          label='Density Estimation')
            
            ymin, ymax = axes[idx].get_ylim()
            axes[idx].set_ylim(ymin, ymax*1.15)
            
            subplot_title_size = 14
            axes[idx].set_title(custom_titles[idx], fontsize=subplot_title_size, pad=5)
            
            stats_text_size = 8
            stats_text = (f'Mean: {weights.mean():.4f}   Std: {weights.std():.4f}\n'
                         f'Min: {weights.min():.4f}   Max: {weights.max():.4f}')
            axes[idx].text(0.97, 0.92, stats_text,
                          transform=axes[idx].transAxes,
                          horizontalalignment='right',
                          verticalalignment='top',
                          bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5),
                          fontsize=stats_text_size)
            
            y_tick_label_size = 10
            axes[idx].tick_params(axis='y', labelsize=y_tick_label_size)
            
            legend_font_size = 10
            axes[idx].legend(fontsize=legend_font_size, 
                            frameon=True, 
                            framealpha=0.8, 
                            edgecolor='black',
                            loc='upper left',
                            handlelength=0.7,
                            handletextpad=0.3)
            
            axes[idx].grid(True, linestyle='--', alpha=0.9, linewidth=0.5)
    
    x_tick_label_size = 10
    plt.tick_params(axis='x', labelsize=x_tick_label_size)
    
    axis_label_size = 12
    fig.text(0.5, 0.01, 'Weight Value', ha='center', fontsize=axis_label_size)
    fig.text(0.01, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=axis_label_size)
    
    plt.subplots_adjust(hspace=0.15, top=0.97, bottom=0.07, left=0.1, right=0.98)
    
    plt.savefig(os.path.join(vis_dir, 'weights_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved compact weight distribution plot without main title")

#------------------------------------------------------------------------------
# Attack Comparison Visualization
#------------------------------------------------------------------------------
def plot_attack_comparison(base_path, vis_dir):
    methods = ['nobias_backprop', 'nobias_biprop', 'nobias_halfbiprop']
    classic_colors = ['#4878d0', '#ee854a', '#d65f5f']
    
    final_values = []
    
    for method in methods:
        base_path_method = os.path.join(base_path, "csv", f"mnist_nn_{method}")
        acc_path = os.path.join(base_path_method, "ANN_accuracy.csv")
        print(f"Looking for accuracy file at: {acc_path}")
        
        if os.path.exists(acc_path):
            print(f"Found accuracy file for {method}")
            acc_data = np.loadtxt(acc_path, delimiter=',')
            final_clean = acc_data[-1, 1]
            final_noise = acc_data[-1, 2]
            final_fgsm = acc_data[-1, 3]
            
            final_values.append({
                'method': method,
                'clean': final_clean,
                'noise': final_noise,
                'fgsm': final_fgsm
            })
            print(f"Added values for {method}: clean={final_clean}, noise={final_noise}, fgsm={final_fgsm}")
        else:
            print(f"WARNING: Could not find accuracy file for {method}")
    
    if not final_values:
        print("ERROR: No accuracy data was found for any method. Cannot create plot.")
        return
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.edgecolor'] = '#333333'
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    clean_bars = ax.bar(x - width, [v['clean'] for v in final_values], width, 
                         label='Clean', color=classic_colors[0], edgecolor='black', linewidth=0.5)
    noise_bars = ax.bar(x, [v['noise'] for v in final_values], width, 
                          label='Noise', color=classic_colors[1], edgecolor='black', linewidth=0.5)
    fgsm_bars = ax.bar(x + width, [v['fgsm'] for v in final_values], width, 
                         label='FGSM', color=classic_colors[2], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Accuracy', fontsize=18, fontweight='normal')
    ax.set_title('Comparison of Model Performance Under Different Conditions', fontsize=18, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels(['BP no bias', 'BL no bias', 'BL->BP no bias'], fontsize=15)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(frameon=True, framealpha=0.7, fontsize=13, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    
    for bars in [clean_bars, noise_bars, fgsm_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=13, color='#333333')
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'attack_comparison_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved classic attack comparison bar chart")

#------------------------------------------------------------------------------
# Data Loading Functions
#------------------------------------------------------------------------------
def read_summary_file(base_path_method):
    summary_path = os.path.join(base_path_method, "ANN_summary.txt")
    print(f"Looking for summary file at: {summary_path}")
    best_values = {}
    
    if os.path.exists(summary_path):
        print(f"Found summary file")
        with open(summary_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if "Best accuracy test data:" in line:
                best_values['clean_acc'] = float(lines[i+2].split(': ')[1])
                best_values['noise_acc'] = float(lines[i+3].split(': ')[1])
                best_values['fgsm_acc'] = float(lines[i+4].split(': ')[1])
                best_values['sigmoid_rate'] = float(lines[i+5].split(': ')[1])
                best_values['softmax_rate'] = float(lines[i+6].split(': ')[1])
                print(f"Successfully read values: {best_values}")
                break
    else:
        print(f"WARNING: Could not find summary file")
    
    return best_values

#------------------------------------------------------------------------------
# Metrics Comparison Visualization
#------------------------------------------------------------------------------
def create_five_metrics_comparison(base_path, vis_dir, all_data, methods):
    print("\nStarting five metrics comparison plot...")
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    colors = {
        'nobias_backprop': '#4878d0',
        'nobias_biprop': '#ee854a',
        'nobias_halfbiprop': '#d65f5f'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics = [
        ('clean_acc', 'Clean Data Accuracy'),
        ('noise_acc', 'Noise Robustness'),
        ('fgsm_acc', 'FGSM Robustness'),
        ('sigmoid_clean', 'Sigmoid Activation'),
        ('softmax_clean', 'Softmax Distribution')
    ]
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        
        for method in methods:
            if method in all_data:
                x_data = all_data[method]['iterations']
                y_data = all_data[method][metric]
                
                ax.plot(x_data, y_data,
                       color=colors[method],
                       linewidth=2.0,
                       label=method.replace('nobias_', ''))
        
        ax.set_title(title, fontsize=24, pad=15)
        ax.set_xlabel('Iterations', fontsize=18)
        ax.set_ylabel('Value', fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(-500, 50000)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    axes[-1].remove()
    
    fig.legend(
        [plt.Line2D([0], [0], color=colors[m], lw=2) for m in methods],
        [m.replace('nobias_', '') for m in methods],
        loc='center right',
        bbox_to_anchor=(0.98, 0.5),
        fontsize=14
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'five_metrics_comparison.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print("Saved five metrics comparison plot")

#------------------------------------------------------------------------------
# Main Visualization Function
#------------------------------------------------------------------------------
def create_enhanced_visualizations(base_path):
    vis_dir = os.path.join(base_path, "visualization", "ANN_perfomance_visual")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Created visualization directory at: {vis_dir}")

    methods = ['nobias_backprop', 'nobias_biprop', 'nobias_halfbiprop']
    classic_colors = ['#4878d0', '#ee854a', '#d65f5f']
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    all_data = {}
    
    for method in methods:
        base_path_method = os.path.join(base_path, "csv", f"mnist_nn_{method}")
        print(f"\nProcessing method: {method}")
        print(f"Looking in directory: {base_path_method}")
        
        acc_path = os.path.join(base_path_method, "ANN_accuracy.csv")
        sigmoid_path = os.path.join(base_path_method, "ANN_sigmoid.csv")
        softmax_path = os.path.join(base_path_method, "ANN_softmax.csv")
        
        print(f"Looking for files:")
        print(f"  Accuracy: {acc_path}")
        print(f"  Sigmoid: {sigmoid_path}")
        print(f"  Softmax: {softmax_path}")
        
        if all(os.path.exists(p) for p in [acc_path, sigmoid_path, softmax_path]):
            print(f"Found all required files for {method}")
            acc_data = np.loadtxt(acc_path, delimiter=',')
            sigmoid_data = np.loadtxt(sigmoid_path, delimiter=',')
            softmax_data = np.loadtxt(softmax_path, delimiter=',')
            
            all_data[method] = {
                'iterations': acc_data[:, 0],
                'clean_acc': acc_data[:, 1],
                'noise_acc': acc_data[:, 2],
                'fgsm_acc': acc_data[:, 3],
                'sigmoid_clean': sigmoid_data[:, 1],
                'sigmoid_noise': sigmoid_data[:, 2],
                'sigmoid_fgsm': sigmoid_data[:, 3],
                'softmax_clean': softmax_data[:, 1],
                'softmax_noise': softmax_data[:, 2],
                'softmax_fgsm': softmax_data[:, 3]
            }
            print(f"Successfully loaded all data for {method}")
        else:
            print(f"WARNING: Could not find all required files for {method}")
            continue

    fig = plt.figure(figsize=(14, 17))
    
    ax1 = fig.add_axes([0.1, 0.73, 0.38, 0.23])
    ax2 = fig.add_axes([0.58, 0.73, 0.38, 0.23])
    ax3 = fig.add_axes([0.1, 0.41, 0.38, 0.23])
    ax4 = fig.add_axes([0.58, 0.41, 0.38, 0.23])
    ax5 = fig.add_axes([0.1, 0.13, 0.38, 0.23])
    ax6 = fig.add_axes([0.58, 0.13, 0.38, 0.23])
    ax6.axis('off')
    
    table_method_labels = ["BP\nnobias", "BL\nno bias", "BL->BP\nno bias"]
    legend_method_labels = ["BP no bias", "BL no bias", "BL->BP no bias"]
    
    legend_lines = []
    
    axes_map = {
        (0, 0): ax1,
        (0, 1): ax2,
        (1, 0): ax3,
        (1, 1): ax4,
        (2, 0): ax5,
    }
    
    for i, (subplot_idx, title, y_key) in enumerate([
        ((0, 0), 'Test Data Accuracy', 'clean_acc'),
        ((0, 1), 'Test Data Accuracy With Noise', 'noise_acc'),
        ((1, 0), 'Test Data Accuracy With FGSM', 'fgsm_acc'),
        ((1, 1), 'Sigmoid Activation Ratio', 'sigmoid_rate'),
        ((2, 0), 'Softmax Distribution Ratio', 'softmax_rate')
    ]):
        ax = axes_map[subplot_idx]
        
        for j, method in enumerate(methods):
            if method in all_data:
                if 'rate' in y_key:
                    if y_key == 'sigmoid_rate':
                        y_data = all_data[method]['sigmoid_noise'] / all_data[method]['sigmoid_clean']
                    else:
                        y_data = all_data[method]['softmax_noise'] / all_data[method]['softmax_clean']
                else:
                    y_data = all_data[method][y_key]
                
                line, = ax.plot(all_data[method]['iterations'], y_data,
                             color=classic_colors[j], 
                             linewidth=2.0,
                             marker='o' if i == 0 else 's' if i == 1 else '^',
                             markersize=5,
                             markevery=len(all_data[method]['iterations'])//10)
                
                if i == 0:
                    legend_lines.append(line)
        
        ax.set_title(title, fontsize=24, pad=15)
        
        if subplot_idx[0] == 0:
            ax.set_xlabel('Iterations', fontsize=18, labelpad=4)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        ax.set_ylabel('Accuracy' if 'Accuracy' in title or 'Robustness' in title else 'Ratio', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim(-500, 50000)
        
        if subplot_idx[0] == 0:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        
        ax.grid(True, linestyle='--', alpha=0.99, linewidth=0.8)
    
    ax6.text(0.5, 0.98, 'Performance Summary', 
             horizontalalignment='center',
             fontsize=28,
             transform=ax6.transAxes)
    
    table_data = []
    headers = ['Method', 'Clean\nAcc', 'Noise\nAcc', 'FGSM\nAcc', 'Sigmoid\nRate', 'Softmax\nRate']
    table_data.append(headers)
    
    for j, method in enumerate(methods):
        base_path_method = os.path.join(base_path, "csv", f"mnist_nn_{method}")
        best_values = read_summary_file(base_path_method)
        
        if best_values:
            row = [
                table_method_labels[j],
                f"{best_values['clean_acc']:.4f}",
                f"{best_values['noise_acc']:.4f}",
                f"{best_values['fgsm_acc']:.4f}",
                f"{best_values['sigmoid_rate']:.4f}",
                f"{best_values['softmax_rate']:.4f}"
            ]
            table_data.append(row)
    
    table = ax6.table(cellText=table_data, loc='center', cellLoc='center', 
                     bbox=[0, 0.25, 1, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 1.1)
    
    for (i, j), cell in table._cells.items():
        if i == 0:
            cell.set_text_props(fontproperties=dict(size=18))
            cell.set_facecolor('#e6e6e6')
        else:
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
    
    legend = ax6.legend(legend_lines, legend_method_labels,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 0.2),
                   frameon=True, 
                   framealpha=0.7, 
                   fontsize=17,
                   edgecolor='black',
                   ncol=3, 
                   columnspacing=0.8,
                   handletextpad=0.3)

    plt.savefig(os.path.join(vis_dir, 'five_metrics_comparison.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print("Saved vertically oriented metrics comparison plot with custom spacing")

    plot_weights_comparison(base_path, vis_dir)
    plot_attack_comparison(base_path, vis_dir)

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    base_path = os.getcwd()
    create_enhanced_visualizations(base_path)