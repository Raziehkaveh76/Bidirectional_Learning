"""
Bidirectional CNN-GAN Implementation for MNIST
============================================

This script implements a bidirectional CNN-GAN architecture for MNIST digit classification
and generation. The model can be configured to use different training modes:
- Standard backpropagation
- Full bidirectional propagation
- Half bidirectional propagation
Each mode can be run with or without bias terms.

Key Features:
- Classifier: CNN for digit recognition
- Generator: Reverse CNN for digit generation
- Discriminator: CNN for real/fake image discrimination
- FGSM attack evaluation
- Extensive logging and visualization
"""

import os
import sys
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

tf.experimental.numpy.experimental_enable_numpy_behavior()

print("TensorFlow version:", tf.__version__)

#-----------------------------------------------------------------------------#
# Helper Functions
#-----------------------------------------------------------------------------#

def print_custom_summary(accuracy_list, sigmoid_list, softmax_list, output_file, batch_size=50):
    """Generate training summary with best metrics."""
    acc_array = np.array(accuracy_list)
    sig_array = np.array(sigmoid_list)
    soft_array = np.array(softmax_list)
    total_samples = 60000
    
    with open(output_file, 'w') as f:
        # Best accuracy on test data
        best_test_idx = np.argmax(acc_array[:, 1])
        f.write("Best accuracy test data:\n")
        f.write(f"Index: {acc_array[best_test_idx, 4]} Iteration: {int(acc_array[best_test_idx, 0])} "
                f"Epoch: {int(acc_array[best_test_idx, 0] * batch_size / total_samples)}\n")
        f.write(f"Test data: {acc_array[best_test_idx, 1]:.4f}\n")
        f.write(f"Test data with noise: {acc_array[best_test_idx, 2]:.4f}\n")
        f.write(f"Test data with FGSM: {acc_array[best_test_idx, 3]:.4f}\n")
        f.write(f"Sigmoid rate of maximum output of noise over real: "
                f"{sig_array[best_test_idx, 2]/sig_array[best_test_idx, 1]:.4f}\n")
        f.write(f"Softmax rate of maximum output of noise over real: "
                f"{soft_array[best_test_idx, 2]/soft_array[best_test_idx, 1]:.4f}\n\n")
        
        # Best accuracy on noisy test data
        best_noise_idx = np.argmax(acc_array[:, 2])
        f.write("Best accuracy test data with noise:\n")
        f.write(f"Index: {acc_array[best_noise_idx, 4]} Iteration: {int(acc_array[best_noise_idx, 0])} "
                f"Epoch: {int(acc_array[best_noise_idx, 0] * batch_size / total_samples)}\n")
        f.write(f"Test data: {acc_array[best_noise_idx, 1]:.4f}\n")
        f.write(f"Test data with noise: {acc_array[best_noise_idx, 2]:.4f}\n")
        f.write(f"Test data with FGSM: {acc_array[best_noise_idx, 3]:.4f}\n")
        f.write(f"Sigmoid rate of maximum output of noise over real: "
                f"{sig_array[best_noise_idx, 2]/sig_array[best_noise_idx, 1]:.4f}\n")
        f.write(f"Softmax rate of maximum output of noise over real: "
                f"{soft_array[best_noise_idx, 2]/soft_array[best_noise_idx, 1]:.4f}\n\n")
        
        # Best accuracy on FGSM test data
        best_fgsm_idx = np.argmax(acc_array[:, 3])
        f.write("Best accuracy test data with FGSM:\n")
        f.write(f"Index: {acc_array[best_fgsm_idx, 4]} Iteration: {int(acc_array[best_fgsm_idx, 0])} "
                f"Epoch: {int(acc_array[best_fgsm_idx, 0] * batch_size / total_samples)}\n")
        f.write(f"Test data: {acc_array[best_fgsm_idx, 1]:.4f}\n")
        f.write(f"Test data with noise: {acc_array[best_fgsm_idx, 2]:.4f}\n")
        f.write(f"Test data with FGSM: {acc_array[best_fgsm_idx, 3]:.4f}\n")
        f.write(f"Sigmoid rate of maximum output of noise over real: "
                f"{sig_array[best_fgsm_idx, 2]/sig_array[best_fgsm_idx, 1]:.4f}\n")
        f.write(f"Softmax rate of maximum output of noise over real: "
                f"{soft_array[best_fgsm_idx, 2]/soft_array[best_fgsm_idx, 1]:.4f}\n\n")
        
        # Best sigmoid rate
        sigmoid_rates = sig_array[:, 2] / sig_array[:, 1]
        best_sig_idx = np.argmax(sigmoid_rates)
        f.write(f"Best sigmoid rate of maximum output of noise over real: {sigmoid_rates[best_sig_idx]:.4f}\n")
        f.write(f"Index: {acc_array[best_sig_idx, 4]} Iteration: {int(acc_array[best_sig_idx, 0])} "
                f"Epoch: {int(acc_array[best_sig_idx, 0] * batch_size / total_samples)}\n")
        f.write("Accuracy:\n")
        f.write(f"Test data: {acc_array[best_sig_idx, 1]:.4f}\n")
        f.write(f"Test data with noise: {acc_array[best_sig_idx, 2]:.4f}\n")
        f.write(f"Test data with FGSM: {acc_array[best_sig_idx, 3]:.4f}\n\n")
        
        # Best softmax rate
        softmax_rates = soft_array[:, 2] / soft_array[:, 1]
        best_soft_idx = np.argmax(softmax_rates)
        f.write(f"Best softmax rate of maximum output of noise over real: {softmax_rates[best_soft_idx]:.8f}\n")
        f.write(f"Index: {acc_array[best_soft_idx, 4]} Iteration: {int(acc_array[best_soft_idx, 0])} "
                f"Epoch: {int(acc_array[best_soft_idx, 0] * batch_size / total_samples)}\n")
        f.write("Accuracy:\n")
        f.write(f"Test data: {acc_array[best_soft_idx, 1]:.4f}\n")
        f.write(f"Test data with noise: {acc_array[best_soft_idx, 2]:.4f}\n")
        f.write(f"Test data with FGSM: {acc_array[best_soft_idx, 3]:.4f}\n")

def save_test_subset_data(model, X_test, y_test, folder_csv):
    """Save test subset data for digits {2,3,4} and their perturbed versions.
    
    Creates and saves:
    - Clean test images for digits 2,3,4
    - Gaussian noise perturbed versions
    - FGSM adversarial examples
    - Corresponding labels
    
    Args:
        model: The trained model for generating FGSM examples
        X_test: Full test image set
        y_test: Full test label set
        folder_csv: Directory to save the data files
    """
    # Get indices for digits 2,3,4
    indices_234 = np.where(np.isin(y_test, [2, 3, 4]))[0]
    X_test_234 = tf.cast(X_test[indices_234], tf.float32)
    y_test_234 = y_test[indices_234]
    
    # Add noise
    noise = tf.random.normal(X_test_234.shape, mean=0.0, stddev=0.1, dtype=tf.float32)
    X_test_234_noise = tf.clip_by_value(X_test_234 + noise, 0.0, 1.0)
    
    # Generate FGSM examples
    fgsm_params = {'eps': 0.3, 'norm': np.inf, 'clip_min': 0., 'clip_max': 1.}
    _, X_test_234_fgsm = evaluate_fgsm(model, X_test_234, y_test_234, fgsm_params)
    
    # Save all versions with consistent names
    np.save(os.path.join(folder_csv, "cnn_test_subset_234_images.npy"), X_test_234.reshape(-1, 784))
    np.save(os.path.join(folder_csv, "cnn_test_subset_234_noise.npy"), X_test_234_noise.reshape(-1, 784))
    np.save(os.path.join(folder_csv, "cnn_test_subset_234_fgsm.npy"), X_test_234_fgsm.reshape(-1, 784))
    np.save(os.path.join(folder_csv, "cnn_test_subset_234_labels.npy"), y_test_234)

def save_results(model, mnist_data, folder_csv, folder_out, weights_subfolder,
                 accuracy_list, sigmoid_list, softmax_list, folder_Weights):
    """Save comprehensive training results and model artifacts.
    
    Saves:
    1. CSV logs (accuracy, sigmoid, softmax metrics)
    2. Performance summary text file
    3. Model weights
    4. Test subset data (clean, noisy, adversarial)
    5. Visualization data (logits, hidden states)
    6. Verification outputs
    
    Args:
        model: Trained model to save
        mnist_data: MNIST dataset object
        folder_csv: Directory for CSV logs
        folder_out: Directory for output files
        weights_subfolder: Subdirectory name for weights
        accuracy_list: List of accuracy metrics
        sigmoid_list: List of sigmoid metrics
        softmax_list: List of softmax metrics
        folder_Weights: Base directory for weights
    """
    # 1) Write CSV logs with consistent names
    with open(os.path.join(folder_csv, "cnn_accuracy.csv"), "w", newline='') as f:
        csv.writer(f).writerows(accuracy_list)
    with open(os.path.join(folder_csv, "cnn_sigmoid.csv"), "w", newline='') as f:
        csv.writer(f).writerows(sigmoid_list)
    with open(os.path.join(folder_csv, "cnn_softmax.csv"), "w", newline='') as f:
        csv.writer(f).writerows(softmax_list)

    # 2) Summaries
    summary_path = os.path.join(folder_csv, "cnn_summary.txt")
    print_custom_summary(accuracy_list, sigmoid_list, softmax_list, summary_path, batch_size=50)
    print(f"Saved summary to: {summary_path}")

    # 3) Save weights with consistent subfolder name
    weights_folder = os.path.join(folder_Weights, weights_subfolder)
    os.makedirs(weights_folder, exist_ok=True)
    
    np.save(os.path.join(weights_folder, "C_W1.npy"), model.layer.C_W1.numpy())
    np.save(os.path.join(weights_folder, "C_W2.npy"), model.layer.C_W2.numpy())
    np.save(os.path.join(weights_folder, "C_W3.npy"), model.layer.C_W3.numpy())
    np.save(os.path.join(weights_folder, "C_W4.npy"), model.layer.C_W4.numpy())
    
    if model.layer.use_bias:  # Use property
        np.save(os.path.join(weights_folder, "C_B1.npy"), model.layer.C_B1.numpy())
        np.save(os.path.join(weights_folder, "C_B2.npy"), model.layer.C_B2.numpy())
        np.save(os.path.join(weights_folder, "C_B3.npy"), model.layer.C_B3.numpy())
        np.save(os.path.join(weights_folder, "C_B4.npy"), model.layer.C_B4.numpy())
    
    print(f"Saved CNN weights to: {weights_folder}")

    # 4) Get 2,3,4 test subset first
    indices_234 = np.where(np.isin(np.argmax(mnist_data.test.labels, axis=1), [2, 3, 4]))[0]
    X_test_234 = tf.cast(mnist_data.test.images[indices_234], tf.float32)
    y_test_234 = np.argmax(mnist_data.test.labels[indices_234], axis=1)

    # 5) Create the new subdirectory in the output folder
    subsets_dir = os.path.join(folder_out, "subsets")
    os.makedirs(subsets_dir, exist_ok=True)

    # 6) Now create noise and FGSM versions
    noise = tf.random.uniform(X_test_234.shape, -0.1, 0.1)
    noisy_images = tf.clip_by_value(X_test_234 + noise, 0.0, 1.0)
    fgsm_params = {'eps': 0.3, 'norm': np.inf, 'clip_min': 0., 'clip_max': 1.}
    adv_examples = fast_gradient_method(model, X_test_234, **fgsm_params)
    
    # IMPORTANT: Save files needed for visualization script to CSV folder
    np.save(os.path.join(folder_csv, "cnn_test_subset_234_images.npy"), X_test_234.reshape(-1, 784))
    np.save(os.path.join(folder_csv, "cnn_test_subset_234_labels.npy"), y_test_234)
    np.save(os.path.join(folder_csv, "cnn_test_subset_234_noise.npy"), noisy_images.numpy().reshape(-1, 784))
    np.save(os.path.join(folder_csv, "cnn_test_subset_234_fgsm.npy"), adv_examples.numpy().reshape(-1, 784))
    
    # 7) Save additional data to the new directory structure
    # Save raw tensor format
    np.save(os.path.join(subsets_dir, 'X_test_234.npy'), X_test_234)
    np.save(os.path.join(subsets_dir, 'y_test_234.npy'), y_test_234)
    
    # Save logits for visualization
    viz_logits_clean = model.get_visualization_logits(X_test_234, training=False)
    viz_logits_noise = model.get_visualization_logits(tf.reshape(noisy_images, [-1, 28, 28, 1]), training=False) 
    viz_logits_fgsm = model.get_visualization_logits(adv_examples, training=False)
    
    np.save(os.path.join(subsets_dir, "viz_logits_clean.npy"), viz_logits_clean)
    np.save(os.path.join(subsets_dir, "viz_logits_noise.npy"), viz_logits_noise) 
    np.save(os.path.join(subsets_dir, "viz_logits_fgsm.npy"), viz_logits_fgsm)
    
    # 8) Save second 25 adversarial examples
    second_25_test = mnist_data.test.images[25:50]
    second_25_labels = mnist_data.test.labels[25:50]
    _, second_adv_examples = evaluate_fgsm(model, second_25_test, second_25_labels, fgsm_params)
    
    # Save these examples
    np.save(os.path.join(subsets_dir, "second_25_adv_examples.npy"), second_adv_examples.numpy())
    
    # 9) Visualize hidden layer and save FGSM examples
    visualize_hidden_layer(model, mnist_data, folder_out, "final", start_idx=25)
    
    fig = plot_generator(second_adv_examples, figsize=(10, 10), grid_size=(5, 5))
    plt.savefig(os.path.join(folder_out, "adv_second_final.png"), bbox_inches='tight')
    plt.close(fig)
    
    # 10) Verification test for data format and probability outputs
    print("\n=== Data Format and Probability Verification ===")
    print(f"X_test_234 shape: {X_test_234.shape}")
    print(f"Saved shape: {X_test_234.reshape(-1, 784).shape}")

    # Test forward pass on a few samples
    print("\nTesting forward pass:")
    softmax, sigmoid, logits = model.classifier(X_test_234[:3].reshape(-1, 28, 28, 1), training=False)
    print(f"Logits shape: {logits.shape}")
    print(f"Sample logits (digits 2,3,4):")
    print(logits[:3, 2:5].numpy())

    print("\nProbability Output Verification:")
    print(f"Softmax outputs (sum should be 1.0):")
    print(softmax[:3].numpy())
    print(f"Softmax sums: {np.sum(softmax[:3].numpy(), axis=1)}")
    print(f"Max softmax value: {np.max(softmax[:3].numpy())}")

    print(f"\nSigmoid outputs (independent probabilities):")
    print(sigmoid[:3].numpy())
    print(f"Max sigmoid value: {np.max(sigmoid[:3].numpy())}")

    # Test visualization logits - should be same as regular logits
    print("\nComparing regular vs visualization logits:")
    viz_logits = model.get_visualization_logits(X_test_234[:3].reshape(-1, 28, 28, 1), training=False)
    print(f"Regular logits: {logits[:3, 2:5].numpy()}")
    print(f"Viz logits: {viz_logits[:3, 2:5].numpy()}")
    print(f"Are identical: {np.allclose(logits, viz_logits)}")

#-----------------------------------------------------------------------------#
# Bidirectional CNN Model
#-----------------------------------------------------------------------------#

class CNNBidirectionalLayer(tf.keras.layers.Layer):
    """Core layer implementing bidirectional CNN architecture."""
    def __init__(self, config_num, name=None):
        super(CNNBidirectionalLayer, self).__init__(name=name)
        self.k_filters = 64
        self.l_filters = 128
        self.m_units = 1024
        self.z_dim = 10
        self.config_num = config_num
        
        # Create BN layers
        self.c_bn2 = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            center=True,
            scale=True,
            name='c_bn2'
        )
        
        self.g_bn2 = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            center=True,
            scale=True,
            name='g_bn2'
        )
        
        self.g_bn3 = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            center=True,
            scale=True,
            name='g_bn3'
        )

        self.d_bn2 = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            center=True,
            scale=True,
            name='d_bn2'
        )

    @property
    def use_bias(self):
        return self.config_num < 3
    
    @property
    def is_bidirectional(self):
        return self.config_num in [1, 2, 4, 5]
    
    @property
    def is_half_bidirectional(self):
        return self.config_num in [2, 5]
        
    def _get_temperature(self, for_visualization=False):
        """Always return 1.0"""
        return 1.0
            
    def should_train_generator(self, step, max_half_steps=5000):
        """For half-biprop, train generator up to 750k steps."""
        if self.config_num in [1, 4]:  # Full biprop
            return True
        elif self.config_num in [2, 5] and step < max_half_steps:
            return True
        return False

    def build(self, input_shape):
        # Classifier weights (shared with generator)
        self.C_W1 = self.add_weight(
            name="C_W1",
            shape=(4, 4, 1, self.k_filters),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)
        )
        self.C_W2 = self.add_weight(
            name="C_W2",
            shape=(4, 4, self.k_filters, self.l_filters),
            initializer="glorot_uniform"
        )
        self.C_W3 = self.add_weight(
            name="C_W3",
            shape=(7 * 7 * self.l_filters, self.m_units),
            initializer="glorot_uniform"
        )
        self.C_W4 = self.add_weight(
            name="C_W4",
            shape=(self.m_units, self.z_dim),
            initializer="glorot_uniform"
        )

        if self.use_bias:
            self.C_B1 = self.add_weight(
                name="C_B1",
                shape=[self.k_filters],
                initializer="zeros"
            )
            self.C_B2 = self.add_weight(
                name="C_B2",
                shape=[self.l_filters],
                initializer="zeros"
            )
            self.C_B3 = self.add_weight(
                name="C_B3",
                shape=[self.m_units],
                initializer="zeros"
            )
            self.C_B4 = self.add_weight(
                name="C_B4",
                shape=[self.z_dim],
                initializer="zeros"
            )

            self.G_B1 = self.add_weight(
                name="G_B1",
                shape=[1],
                initializer="zeros"
            )
            self.G_B2 = self.add_weight(
                name="G_B2",
                shape=[self.k_filters],
                initializer="zeros"
            )
            self.G_B3 = self.add_weight(
                name="G_B3",
                shape=[7 * 7 * self.l_filters],
                initializer="zeros"
            )
            self.G_B4 = self.add_weight(
                name="G_B4",
                shape=[self.m_units],
                initializer="zeros"
            )

        # Discriminator weights
        self.D_W1 = self.add_weight(
            name="D_W1",
            shape=(4, 4, 1, self.k_filters),
            initializer="glorot_uniform",
            dtype=tf.float32
        )
        self.D_W2 = self.add_weight(
            name="D_W2",
            shape=(4, 4, self.k_filters, self.l_filters),
            initializer="glorot_uniform",
            dtype=tf.float32
        )
        self.D_W3 = self.add_weight(
            name="D_W3",
            shape=(7 * 7 * self.l_filters, self.m_units),
            initializer="glorot_uniform",
            dtype=tf.float32
        )
        self.D_W4 = self.add_weight(
            name="D_W4",
            shape=(self.m_units, 1),
            initializer="glorot_uniform",
            dtype=tf.float32
        )

        self.D_B1 = self.add_weight(
            name="D_B1",
            shape=[self.k_filters],
            initializer="zeros",
            dtype=tf.float32
        )
        self.D_B2 = self.add_weight(
            name="D_B2",
            shape=[self.l_filters],
            initializer="zeros",
            dtype=tf.float32
        )
        self.D_B3 = self.add_weight(
            name="D_B3",
            shape=[self.m_units],
            initializer="zeros",
            dtype=tf.float32
        )
        self.D_B4 = self.add_weight(
            name="D_B4",
            shape=[1],
            initializer="zeros",
            dtype=tf.float32
        )

        # Initialize BN layers by calling them once
        dummy_shape2 = (1, 7, 7, self.l_filters)
        dummy_shape3 = (1, self.m_units)
        dummy_input2 = tf.zeros(dummy_shape2)
        dummy_input3 = tf.zeros(dummy_shape3)
        
        # Force initialization of BN layers
        self.g_bn2(dummy_input2, training=True)
        self.g_bn3(dummy_input3, training=True)

    def forward(self, x, training=True):
        """Classifier forward pass with temperature scaling."""
        h1 = tf.nn.conv2d(x, self.C_W1, strides=[1,2,2,1], padding='SAME')
        if self.use_bias:
            h1 = tf.nn.bias_add(h1, self.C_B1)
        h1 = tf.nn.leaky_relu(h1, alpha=0.2)
        
        h2 = tf.nn.conv2d(h1, self.C_W2, strides=[1,2,2,1], padding='SAME')
        if self.use_bias:
            h2 = tf.nn.bias_add(h2, self.C_B2)
        h2 = self.c_bn2(h2, training=training)
        h2 = tf.nn.leaky_relu(h2, alpha=0.2)
        
        h2_flat = tf.reshape(h2, [-1, 7*7*self.l_filters])
        h3 = tf.matmul(h2_flat, self.C_W3)
        if self.use_bias:
            h3 = tf.nn.bias_add(h3, self.C_B3)
        h3 = tf.nn.relu(h3)
        
        logits = tf.matmul(h3, self.C_W4)
        if self.use_bias:
            logits = tf.nn.bias_add(logits, self.C_B4)
        
        # Direct calculation without temperature
        return tf.nn.softmax(logits), tf.nn.sigmoid(logits), logits

    def discriminator(self, x, training=True):
        """Discriminator pass => returns (sigmoidOut, logits)."""
        x = tf.reshape(x,[-1,28,28,1])
        h1= tf.nn.conv2d(x,self.D_W1,strides=[1,2,2,1],padding='SAME')
        h1= tf.nn.bias_add(h1,self.D_B1)
        h1= tf.nn.leaky_relu(h1, alpha=0.2)

        h2= tf.nn.conv2d(h1,self.D_W2,strides=[1,2,2,1],padding='SAME')
        h2= tf.nn.bias_add(h2,self.D_B2)
        h2 = self.d_bn2(h2, training=training)
        h2 = tf.nn.leaky_relu(h2, alpha=0.2)

        h2= tf.reshape(h2,[-1,7*7*self.l_filters])
        h3= tf.matmul(h2, self.D_W3) + self.D_B3
        h3= tf.nn.leaky_relu(h3, alpha=0.2)

        logits= tf.matmul(h3, self.D_W4) + self.D_B4
        out= tf.nn.sigmoid(logits)
        return out, logits

    def backward(self, z, batch_size, training=True):
        """Generator backward pass WITHOUT stopping gradients, matching TF1.x"""
        z = tf.cast(z, tf.float32)
        
        # Dense1: matmul -> bias -> BN -> relu
        h3 = tf.matmul(z, tf.transpose(self.C_W4))
        if self.use_bias:
            h3 = tf.nn.bias_add(h3, self.G_B4)
        h3 = self.g_bn3(h3, training=training)
        h3 = tf.nn.relu(h3)

        # Dense2: matmul -> bias -> reshape -> BN -> relu
        h2 = tf.matmul(h3, tf.transpose(self.C_W3))
        if self.use_bias:
            h2 = tf.nn.bias_add(h2, self.G_B3)
        h2 = tf.reshape(h2, [batch_size, 7, 7, 128])
        h2 = self.g_bn2(h2, training=training)
        h2 = tf.nn.relu(h2)

        h1 = tf.nn.conv2d_transpose(h2, self.C_W2, 
                                   output_shape=[batch_size,14,14, self.k_filters],
                                   strides=[1,2,2,1], padding='SAME')
        if self.use_bias:
            h1 = tf.nn.bias_add(h1, self.G_B2)
        h1 = tf.nn.relu(h1)

        logits = tf.nn.conv2d_transpose(h1, self.C_W1,
            output_shape=[batch_size,28,28,1],
            strides=[1,2,2,1], padding='SAME')
        if self.use_bias:
            logits = tf.nn.bias_add(logits, self.G_B1)
        
        return tf.nn.sigmoid(logits), logits

    def get_visualization_logits(self, x, training=False):
        _, _, logits = self.forward(x, training=training)
        return logits

class CNNBidirectionalNetwork(tf.keras.Model):
    def __init__(self, config_num):
        super().__init__()
        self.config_num= config_num
        self.layer      = CNNBidirectionalLayer(config_num)
        self.layer.build((None,28,28,1))

        self.classifier_vars    = []
        self.generator_vars     = []
        self.discriminator_vars = []
        self._setup_var_lists()

    def _setup_var_lists(self):
        t_vars = self.trainable_variables
        
        print("\nAll trainable variables:")
        for var in t_vars:
            print(f"  {var.name}")
        
        # Classifier vars
        self.classifier_vars = [var for var in t_vars if 'C_' in var.name or 'c_bn' in var.name]
        if not self.layer.use_bias:
            self.classifier_vars = [var for var in t_vars if 'C_W' in var.name or 'c_bn' in var.name]
        
        # Generator vars: Include classifier weights for biprop configs
        if self.layer.is_bidirectional:
            self.generator_vars = [var for var in t_vars if 'C_W' in var.name or 'g_bn' in var.name]
            if self.layer.use_bias:
                g_bias_vars = [var for var in t_vars if 'G_B' in var.name]
                self.generator_vars.extend(g_bias_vars)
        else:
            self.generator_vars = []
            g_bn2_vars = self.layer.g_bn2.trainable_variables
            g_bn3_vars = self.layer.g_bn3.trainable_variables
            self.generator_vars.extend(g_bn2_vars)
            self.generator_vars.extend(g_bn3_vars)
            if self.layer.use_bias:
                g_bias_vars = [var for var in t_vars if 'G_B' in var.name]
                self.generator_vars.extend(g_bias_vars)
        
        # Discriminator vars
        self.discriminator_vars = [var for var in t_vars if 'D_' in var.name]
        
        print("\nVariable assignments:")
        print(f"Generator variables ({len(self.generator_vars)}):")
        for var in self.generator_vars:
            print(f"  {var.name}")
        print("\nClassifier variables:")
        for var in self.classifier_vars:
            print(f"  {var.name}")
        print("\nDiscriminator variables:")
        for var in self.discriminator_vars:
            print(f"  {var.name}")

    def classifier(self, x, training=True):
        return self.layer.forward(x, training=training)
    def generator(self, z, batch_size, training=True):
        return self.layer.backward(z, batch_size, training=training)
    def discriminator(self, x, training=True):
        return self.layer.discriminator(x, training=training)
    
    def call(self, x, training=False):
        return self.layer.forward(x, training=training)[2]

    def get_visualization_logits(self, x, training=False):
        return self.layer.get_visualization_logits(x, training=training)

#-----------------------------------------------------------------------------#
# Data Classes
#-----------------------------------------------------------------------------#

class MNISTDataset:
    """MNIST dataset wrapper with batching."""
    def __init__(self, images, labels):
        self.images = images.astype('float32')
        self.labels = labels.astype('float32')
        self.num_examples = images.shape[0]
        self._index = 0
# to produce mini batches with optional shuffling       
    def next_batch(self, batch_size):
        if self._index + batch_size> self.num_examples:
            perm = np.random.permutation(self.num_examples)
            self.images= self.images[perm]
            self.labels= self.labels[perm]
            self._index= 0
        st= self._index
        self._index+= batch_size
        return self.images[st:self._index], self.labels[st:self._index]

# load the data reshape to 28x28x1 and normalize to [0,1]
# convert labels to one-hot encoding
class MNISTData:
    def __init__(self):
        mnist= tf.keras.datasets.mnist
        (train_x, train_y), (test_x, test_y)= mnist.load_data()

        train_x= train_x.reshape(-1,28,28,1).astype('float32')/255.0
        test_x = test_x.reshape(-1,28,28,1).astype('float32')/255.0
        train_y= tf.keras.utils.to_categorical(train_y,10).astype('float32')
        test_y = tf.keras.utils.to_categorical(test_y,10).astype('float32')

        self.train= MNISTDataset(train_x, train_y)
        self.test = MNISTDataset(test_x,  test_y)

#-----------------------------------------------------------------------------#
# Training Helpers
#-----------------------------------------------------------------------------#

def sample_Z(m, n):
    """Generate uniform random noise [-1,1]."""
    return np.random.uniform(-1.,1., size=[m,n]).astype(np.float32)

@tf.function
def train_classifier_step(model, x, y, optimizer, training=True):
    """Execute one classifier training step."""
    with tf.GradientTape() as tape:
        _, _, logits = model.classifier(x, training=training)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        )
    grads = tape.gradient(loss, model.classifier_vars)
    optimizer.apply_gradients(zip(grads, model.classifier_vars))
    return loss

@tf.function
def train_discriminator_step(model, real_images, batch_size, optimizer):
    """Execute one training step for the discriminator.
    
    Trains discriminator to distinguish between real and generated images.
    Uses binary cross-entropy loss.
    
    Args:
        model: Model being trained
        real_images: Batch of real MNIST images
        batch_size: Number of images per batch
        optimizer: Optimizer for weight updates
    
    Returns:
        Total discriminator loss
    """
    z= sample_Z(batch_size, 10)
    fake_images, _= model.generator(z, batch_size, training=True)
    with tf.GradientTape() as tape:
        real_out, real_logits= model.discriminator(real_images, training=True)
        fake_out, fake_logits= model.discriminator(fake_images, training=True)
        real_loss= tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_logits),
                logits=real_logits
            )
        )
        fake_loss= tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_logits),
                logits=fake_logits
            )
        )
        total_loss= real_loss+ fake_loss
    grads= tape.gradient(total_loss, model.discriminator_vars)
    optimizer.apply_gradients(zip(grads, model.discriminator_vars))
    return total_loss

@tf.function
def train_generator_step(model, batch_size, optimizer):
    """Execute one training step for the generator.
    
    Trains generator to produce images that fool the discriminator.
    Uses binary cross-entropy loss with "real" labels.
    
    Args:
        model: Model being trained
        batch_size: Number of images to generate
        optimizer: Optimizer for weight updates
    
    Returns:
        Generator loss value
    """
    z = sample_Z(batch_size, 10)
    with tf.GradientTape() as tape:
        fake_images, _ = model.generator(z, batch_size, training=True)
        fake_out, fake_logits = model.discriminator(fake_images, training=True)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_logits),
                logits=fake_logits
            )
        )
    grads = tape.gradient(loss, model.generator_vars)
    
    
    # grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    
    optimizer.apply_gradients(zip(grads, model.generator_vars))
    return loss

def plot_generator(samples, figsize=(10, 10), grid_size=(5, 5)):
    """Create a grid visualization of generated samples.
    
    Args:
        samples: Tensor or array of images to plot
        figsize: Figure dimensions (width, height)
        grid_size: Grid dimensions (rows, cols)
    
    Returns:
        matplotlib Figure object
    """
    rows, cols = grid_size
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.05, hspace=0.05)
    if isinstance(samples, tf.Tensor):
        samples = samples.numpy()
    
    max_samples = rows * cols
    for i, sample in enumerate(samples[:max_samples]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')
    return fig

class ClassifierModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model= model
    def get_logits(self, x):
        _, _, logits= self.model.classifier(x, training=False)
        return logits

def evaluate_fgsm(model, images, labels, fgsm_params):
    """Generate and evaluate FGSM adversarial examples.
    
    Creates adversarial examples using the Fast Gradient Sign Method
    and evaluates model accuracy on them.
    
    Args:
        model: Model to attack/evaluate
        images: Clean input images
        labels: True labels
        fgsm_params: Dictionary with attack parameters
            - eps: Maximum perturbation
            - norm: Distance norm
            - clip_min/max: Value range for valid images
    
    Returns:
        tuple: (accuracy on adversarial examples, adversarial examples)
    """
    x= tf.convert_to_tensor(images)
    y= tf.convert_to_tensor(labels)
    with tf.GradientTape() as tape:
        tape.watch(x)
        _, _, logits= model.classifier(x, training=False)
        loss= tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    grads= tape.gradient(loss, x)
    signed_grad= tf.sign(grads)
    eps= fgsm_params.get('eps', 0.3)
    adv_x= x+ eps * signed_grad
    adv_x= tf.clip_by_value(adv_x, 0.0,1.0)

    softmax_out, _, _= model.classifier(adv_x, training=False)
    accuracy= tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(softmax_out, axis=1),
                         tf.argmax(y, axis=1)),
                tf.float32)
    )
    return accuracy, adv_x

def visualize_hidden_layer(model, mnist_data, folder_out, step, start_idx=0):
    """Visualize activations of the second convolutional layer.
    
    Creates and saves visualization of mean activations across channels
    for a batch of test images.
    
    Args:
        model: Model to visualize
        mnist_data: MNIST dataset object
        folder_out: Directory to save visualizations
        step: Current training step (for filename)
        start_idx: Starting index in test set
    
    Returns:
        Tensor of layer activations
    """
    test_examples = mnist_data.test.images[start_idx:start_idx+25]
    
    h1 = tf.nn.conv2d(test_examples, model.layer.C_W1, strides=[1,2,2,1], padding='SAME')
    if model.layer.use_bias:
        h1 = tf.nn.bias_add(h1, model.layer.C_B1)
    h1 = tf.nn.leaky_relu(h1, alpha=0.2)
    
    h2 = tf.nn.conv2d(h1, model.layer.C_W2, strides=[1,2,2,1], padding='SAME')
    if model.layer.use_bias:
        h2 = tf.nn.bias_add(h2, model.layer.C_B2)
    h2 = model.layer.c_bn2(h2, training=False)
    h2 = tf.nn.leaky_relu(h2, alpha=0.2)
    
    mean_activations = tf.reduce_mean(h2, axis=-1)
    
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i in range(25):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_aspect('equal')
        plt.imshow(mean_activations[i], cmap='gray')
    
    if isinstance(step, str):
        filename = f"hidden_{start_idx}_{step}.png"
    else:
        filename = f"hidden_{start_idx}_{step:06d}.png"
    
    plt.savefig(os.path.join(folder_out, filename), bbox_inches='tight')
    plt.close(fig)
    
    return h2

#-----------------------------------------------------------------------------#
# Main Training
#-----------------------------------------------------------------------------#

def main():
    """Main training loop with evaluation."""
    config_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    config_dict = {
        0: "backprop", 
        1: "biprop", 
        2: "halfbiprop", 
        3: "nobias_backprop",
        4: "nobias_biprop",
        5: "nobias_halfbiprop"
    }
    
    config_name = config_dict[config_num]
    model_prefix = f"cnn_mnist_{config_name}"
    print("Model name:", model_prefix)

    np.random.seed(0)
    tf.random.set_seed(0)

    mnist_data = MNISTData()

    base_path = os.path.dirname(os.path.abspath(__file__))
    
    folder_out = os.path.join(base_path, "out", model_prefix)
    folder_csv = os.path.join(base_path, "csv", model_prefix)
    folder_logs = os.path.join(base_path, "logs", model_prefix)
    folder_Weights = os.path.join(base_path, "Weights")
    weights_subfolder = f"cnn_{config_name}"
    
    os.makedirs(folder_out, exist_ok=True)
    os.makedirs(folder_csv, exist_ok=True)
    os.makedirs(folder_logs, exist_ok=True)
    os.makedirs(folder_Weights, exist_ok=True)

    print("Creating CNNBidirectionalNetwork...")
    model = CNNBidirectionalNetwork(config_num)

    print("\nVariable overview:")
    print(f"  Total trainable: {len(model.trainable_variables)}")
    print(f"  Classifier: {len(model.classifier_vars)}")
    print(f"  Generator: {len(model.generator_vars)}")
    print(f"  Discriminator: {len(model.discriminator_vars)}")

    # Optimizers
    c_optimizer= tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer= tf.keras.optimizers.Adam(learning_rate=0.0005,  beta_1=0.5)
    d_optimizer= tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    summary_writer= tf.summary.create_file_writer(folder_logs)

    batch_size= 100
    training_steps= 10001

    accuracy_list= []
    sigmoid_list = []
    softmax_list = []

    fgsm_params= {'eps':0.3, 'norm':np.inf, 'clip_min':0., 'clip_max':1.}

    print("\nStarting training for config:", config_dict[config_num])
    for step in range(training_steps):
        bx, by = mnist_data.train.next_batch(batch_size)

        # 1) Classifier step
        c_loss = train_classifier_step(model, bx, by, c_optimizer, training=True)

        # 2) Possibly train D/G
        if model.layer.should_train_generator(step):
            d_loss = train_discriminator_step(model, bx, batch_size, d_optimizer)
            g_loss = train_generator_step(model, batch_size, g_optimizer)
        else:
            d_loss = 0.0
            g_loss = 0.0

        # 3) Evaluate periodically
        if step % 5000 == 0 or step == (training_steps - 1):
            print(f"\n=== Evaluation at step {step} ===")
            
            # Generate 25 images
            test_noise = sample_Z(25, 10)
            gen_imgs = model.generator(test_noise, 25, training=False)[0]
            fig = plot_generator(gen_imgs, figsize=(10, 10), grid_size=(5, 5))
            plt.savefig(os.path.join(folder_out, f"cnn_gen_{step:06d}.png"), bbox_inches='tight')
            plt.close(fig)
            
            # Generate fixed 25 images (10 from one-hots + 15 random)
            one_hot_vectors = np.eye(10).astype(np.float32)
            additional_noise = sample_Z(15, 10)
            combined_input = np.concatenate([one_hot_vectors, additional_noise])
            fixed_gen_imgs = model.generator(combined_input, 25, training=False)[0]
            fig = plot_generator(fixed_gen_imgs, figsize=(10, 10), grid_size=(5, 5))
            plt.savefig(os.path.join(folder_out, f"cnn_gen_fixed_{step:06d}.png"), bbox_inches='tight')
            plt.close(fig)
            
            # Generate FGSM adversarial examples (second 25 test)
            second_25_test = mnist_data.test.images[25:50]
            second_25_labels = mnist_data.test.labels[25:50]
            adv_accuracy, adv_examples = evaluate_fgsm(model, second_25_test, second_25_labels, fgsm_params)
            
            # Save these adversarial examples
            fig = plot_generator(adv_examples, figsize=(10, 10), grid_size=(5, 5))
            plt.savefig(os.path.join(folder_out, f"adv_{step:06d}.png"), bbox_inches='tight')
            plt.close(fig)

            # Visualize hidden layer
            visualize_hidden_layer(model, mnist_data, folder_out, step, start_idx=25)

            # Evaluate on test set
            softmax_test,_,_= model.classifier(mnist_data.test.images, training=False)
            pred_test= tf.argmax(softmax_test, axis=1)
            true_test= tf.argmax(mnist_data.test.labels, axis=1)
            test_acc = tf.reduce_mean(tf.cast(tf.equal(pred_test,true_test), tf.float32))

            # Evaluate noise
            noise= tf.random.uniform(mnist_data.test.images.shape, -0.1, 0.1)
            noisy_imgs= tf.clip_by_value(mnist_data.test.images+noise, 0.0,1.0)
            softmax_noise,_,_= model.classifier(noisy_imgs, training=False)
            pred_noise= tf.argmax(softmax_noise, axis=1)
            noise_acc= tf.reduce_mean(tf.cast(tf.equal(pred_noise,true_test), tf.float32))

            # FGSM on entire test set
            adv_acc, adv_imgs= evaluate_fgsm(model, mnist_data.test.images, mnist_data.test.labels, fgsm_params)

            print(f"step={step}, c_loss={c_loss:.4f}, d_loss={d_loss}, g_loss={g_loss}")
            print(f"  TestAcc={test_acc:.4f}, NoiseAcc={noise_acc:.4f}, FGSMAcc={adv_acc:.4f}")

            accuracy_list.append([
                step, float(test_acc), float(noise_acc), float(adv_acc), step//5000
            ])

            sig_test_max = float(tf.reduce_max(model.classifier(mnist_data.test.images,training=False)[1]))
            sig_noise_max= float(tf.reduce_max(model.classifier(noisy_imgs,training=False)[1]))
            sig_adv_max= float(tf.reduce_max(model.classifier(adv_imgs, training=False)[1]))
            sigmoid_list.append([step, sig_test_max, sig_noise_max, sig_adv_max, step//5000])

            soft_test_max= float(tf.reduce_max(softmax_test))
            soft_noise_max= float(tf.reduce_max(softmax_noise))
            soft_adv_max= float(tf.reduce_max(model.classifier(adv_imgs,training=False)[0]))
            softmax_list.append([step, soft_test_max, soft_noise_max, soft_adv_max, step//5000])

            with summary_writer.as_default():
                tf.summary.scalar("loss/classifier", c_loss, step=step)
                tf.summary.scalar("loss/discriminator", d_loss, step=step)
                tf.summary.scalar("loss/generator", g_loss, step=step)
                tf.summary.scalar("accuracy/test", test_acc, step=step)
                tf.summary.scalar("accuracy/noise", noise_acc, step=step)
                tf.summary.scalar("accuracy/fgsm", adv_acc, step=step)

    print("\nTraining complete.")

    # 4) Save final results
    save_results(
        model, mnist_data, folder_csv, folder_out, weights_subfolder,
        accuracy_list, sigmoid_list, softmax_list, folder_Weights
    )

if __name__=="__main__":
    main()