#!/usr/bin/env python3
"""
CNN_performance_visual.py

Generates training performance visualizations for CNN models:
- Plots accuracy curves for Clean/Noise/FGSM test data
- Compares performance across different training configurations:
  * No-bias Backprop
  * No-bias Biprop
  * No-bias Half-biprop
- Saves high-quality plots with consistent styling
"""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Visualization Settings
#------------------------------------------------------------------------------
PLOT_SETTINGS = {
    'TITLE_SIZE': 32,
    'AXIS_LABEL_SIZE': 28,
    'TICK_LABEL_SIZE': 24,
    'LEGEND_SIZE': 24,
    'FIGURE_SIZE': (14, 10),
    'LINE_WIDTH': 2.5,
    'DPI': 300
}

CONFIGS = {
    3: "nobias_backprop",
    4: "nobias_biprop",
    5: "nobias_halfbiprop"
}

#------------------------------------------------------------------------------
# Plotting Functions
#------------------------------------------------------------------------------
def plot_training_curves_from_data(config_name, base_path):
    """
    Creates and saves accuracy vs epoch plots for a specific configuration.
    
    Plots three curves:
    - Clean test data accuracy
    - Noisy test data accuracy
    - FGSM test data accuracy
    """
    # Setup paths
    csv_folder = os.path.join(base_path, "csv", f"cnn_mnist_{config_name}")
    vis_dir = os.path.join(base_path, "visualization", "CNN_performance_visual")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Processing data from: {csv_folder}")
    
    try:
        # Load accuracy data
        accuracy_file = os.path.join(csv_folder, "cnn_accuracy.csv")
        if not os.path.exists(accuracy_file):
            print(f"Error: Accuracy file not found: {accuracy_file}")
            return
            
        accuracy_data = np.loadtxt(accuracy_file, delimiter=',')
        
        # Convert iterations to epochs (batch_size=50)
        acc_epochs = accuracy_data[:, 0] * 50 / 60000
        clean_acc = accuracy_data[:, 1]
        noise_acc = accuracy_data[:, 2]
        fgsm_acc = accuracy_data[:, 3]
        
        # Create accuracy plot
        plt.figure(figsize=PLOT_SETTINGS['FIGURE_SIZE'])
        
        # Plot accuracy curves
        plt.plot(acc_epochs, clean_acc, 'g-', 
                linewidth=PLOT_SETTINGS['LINE_WIDTH'], 
                label='Clean Test Data')
        plt.plot(acc_epochs, noise_acc, 'b-', 
                linewidth=PLOT_SETTINGS['LINE_WIDTH'], 
                label='Noisy Test Data')
        plt.plot(acc_epochs, fgsm_acc, 'r-', 
                linewidth=PLOT_SETTINGS['LINE_WIDTH'], 
                label='FGSM Test Data')
        
        # Configure plot styling
        plt.xlabel('Epoch', 
                  fontsize=PLOT_SETTINGS['AXIS_LABEL_SIZE'], 
                  labelpad=15)
        plt.ylabel('Accuracy', 
                  fontsize=PLOT_SETTINGS['AXIS_LABEL_SIZE'], 
                  labelpad=15)
        plt.title(f'Accuracy vs. Epoch - {config_name}', 
                 fontsize=PLOT_SETTINGS['TITLE_SIZE'], 
                 pad=20)
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=PLOT_SETTINGS['LEGEND_SIZE'],
                  bbox_to_anchor=(0.5, -0.15),
                  loc='upper center',
                  frameon=True,
                  ncol=3)
        
        plt.xticks(fontsize=PLOT_SETTINGS['TICK_LABEL_SIZE'])
        plt.yticks(fontsize=PLOT_SETTINGS['TICK_LABEL_SIZE'])
        plt.tight_layout()
        
        # Save plot
        acc_output_file = os.path.join(vis_dir, f'accuracy_vs_epoch_{config_name}.png')
        plt.savefig(acc_output_file, 
                   dpi=PLOT_SETTINGS['DPI'], 
                   bbox_inches='tight')
        plt.close()
        
        print(f"Generated accuracy plot: {acc_output_file}")
        
    except Exception as e:
        print(f"Error processing data for {config_name}: {str(e)}")

def generate_all_plots(base_path):
    """
    Generates performance plots for all configurations.
    
    Args:
        base_path: Root directory containing the data folders
    """
    print(f"\nStarting plot generation from: {base_path}")
    
    for config_num, config_name in CONFIGS.items():
        print(f"\nProcessing config {config_num}: {config_name}")
        plot_training_curves_from_data(config_name, base_path)

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    generate_all_plots(base_path)
