"""
Bidirectional Neural Network for MNIST
====================================
Simple feedforward neural network with bidirectional weight sharing.
Supports backprop, biprop, and halfbiprop training modes.
"""

import os
import sys
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import time

print("TensorFlow version:", tf.__version__)

#-----------------------------------------------------------------------------#
# Data Handling Classes
#-----------------------------------------------------------------------------#
class MNISTDataset:
    """Dataset wrapper with batching support."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.num_examples = images.shape[0]
        self._index = 0
        
    def next_batch(self, batch_size):
        """Get next batch with optional shuffling."""
        if self._index + batch_size > self.num_examples:
            perm = np.random.permutation(self.num_examples)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            self._index = 0
        start = self._index
        self._index += batch_size
        return self.images[start:self._index], self.labels[start:self._index]

class MNISTData:
    """MNIST data container with train/test split."""
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train = MNISTDataset(train_images, train_labels)
        self.test = MNISTDataset(test_images, test_labels)

#-----------------------------------------------------------------------------#
# Network Classes
#-----------------------------------------------------------------------------#
class BidirectionalLayer(tf.keras.layers.Layer):
    """Core layer implementing bidirectional weight sharing."""
    def __init__(self, units, use_bias=True, **kwargs):
        super(BidirectionalLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            dtype=tf.float32
        )
        
        if self.use_bias:
            self.bias_forward = self.add_weight(
                name="bias_forward",
                shape=(self.units,),
                initializer="zeros",
                dtype=tf.float32
            )
            self.bias_backward = self.add_weight(
                name="bias_backward",
                shape=(input_shape[-1],),
                initializer="zeros",
                dtype=tf.float32
            )
        
        super(BidirectionalLayer, self).build(input_shape)

    def forward(self, x):
        """Forward pass with optional bias."""
        y = tf.matmul(x, self.weight)
        if self.use_bias:
            y = y + self.bias_forward
        return y

    def backward(self, x):
        """Backward pass with optional bias."""
        y = tf.matmul(x, tf.transpose(self.weight))
        if self.use_bias:
            y = y + self.bias_backward
        return y

class BidirectionalNetwork(keras.Model):
    """Network combining forward and backward passes."""
    def __init__(self, config_num):
        super(BidirectionalNetwork, self).__init__()
        self.use_bias = config_num < 3
        self.layer = BidirectionalLayer(
            units=10,
            use_bias=self.use_bias,
            name="bidirectional_layer"
        )
        self.layer.build((None, 784))
    
    def classifier(self, x):
        """Forward classification pass."""
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, 784])
        logits = self.layer.forward(x)
        sigmoid = tf.nn.sigmoid(logits)
        softmax = tf.nn.softmax(logits)
        return softmax, sigmoid, logits
    
    def generator(self, y):
        """Backward generation pass."""
        gx = self.layer.backward(y)
        gx_logits = tf.reshape(gx, [-1, 28, 28, 1])
        gx_sigmoid = tf.nn.sigmoid(gx_logits)
        return gx_sigmoid, gx_logits
    
    def call(self, x):
        return self.classifier(x)[2]

#-----------------------------------------------------------------------------#
# Training Functions
#-----------------------------------------------------------------------------#
@tf.function
def train_classifier_step(model, x, y, optimizer):
    """Execute one classifier training step."""
    with tf.GradientTape() as tape:
        _, _, logits = model.classifier(x)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def train_generator_step(model, x, y, optimizer, generator_weight=1.0):
    """Execute one generator training step."""
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
    with tf.GradientTape() as tape:
        _, gx_logits = model.generator(y)
        loss = generator_weight * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=gx_logits,
                labels=x_reshaped
            )
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

#-----------------------------------------------------------------------------#
# Main Function
#-----------------------------------------------------------------------------#
def main():
    # Add at the start of main()
    start_time = time.time()  # Total runtime
    
    # Configuration
    config_num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    config_name = {
        0: "backprop",
        1: "biprop",
        2: "halfbiprop",
        3: "nobias_backprop",
        4: "nobias_biprop",
        5: "nobias_halfbiprop"
    }

    script_name = os.path.basename(sys.argv[0]).replace(".py", "")
    model_name = f"{script_name}_{config_name[config_num]}"
    print("Model name:", model_name)

    # Set random seeds
    np.random.seed(0)
    tf.random.set_seed(0)

    # Load and prepare MNIST data
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape and normalize
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # Convert data to float32
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    # Create data iterator
    mnist_data = MNISTData(train_images, train_labels, test_images, test_labels)

    # Setup directories
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Create model-specific folders
    folder_out = os.path.join(base_path, "out", f"mnist_nn_{config_name[config_num]}")
    folder_csv = os.path.join(base_path, "csv", f"mnist_nn_{config_name[config_num]}")
    folder_logs = os.path.join(base_path, "logs", f"mnist_nn_{config_name[config_num]}")
    folder_Weights = os.path.join(base_path, "Weights", f"{config_name[config_num]}")
    
    # Create directories
    os.makedirs(folder_out, exist_ok=True)
    os.makedirs(folder_csv, exist_ok=True)
    os.makedirs(folder_logs, exist_ok=True)
    os.makedirs(folder_Weights, exist_ok=True)

    # Create model and optimizers
    model = BidirectionalNetwork(config_num)
    c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

    # TensorBoard writer
    summary_writer = tf.summary.create_file_writer(folder_logs)

    # Training parameters
    batch_size = 100
    training_epochs = 50001

    # Initialize tracking lists
    accuracy_list = []
    sigmoid_list = []
    softmax_list = []
    loss_history = []
    epoch_history = []

    # FGSM parameters
    fgsm_params = {
        'eps': 0.3,
        'norm': np.inf,
        'clip_min': 0.,
        'clip_max': 1.,
        'targeted': False
    }


    print("==== Starting Training ====")
    print(f"Configuration {config_num} => {config_name[config_num]}")
    
    # Add before training loop
    training_start_time = time.time()  # Training specific time

    # Training loop
    for i in range(training_epochs):
        batch_x, batch_y = mnist_data.train.next_batch(batch_size)
        
        # Train classifier
        c_loss = train_classifier_step(model, batch_x, batch_y, c_optimizer)
        
        # Train generator if appropriate
        should_train_generator = (
            config_num in [1, 4] or 
            (config_num in [2, 5] and i < 25000)
        )
        
        if should_train_generator:
            g_loss = train_generator_step(
                model, batch_x, batch_y, g_optimizer, generator_weight=2.0
            )
        else:
            g_loss = 0.0
            
        # Record loss and epoch every 100 iterations
        if i % 100 == 0:
            current_epoch = i * batch_size / mnist_data.train.num_examples
            epoch_history.append(current_epoch)
            loss_history.append(float(c_loss))

        # Evaluation block (every 500 steps)
        if i % 500 == 0 or i == training_epochs - 1:
            print(f"\n=== Evaluation at iteration {i} ===")
            
            # 1. Clean Test Data
            softmax_test, sigmoid_test, _ = model.classifier(mnist_data.test.images)
            test_preds = tf.argmax(softmax_test, axis=1)
            true_labels = tf.argmax(mnist_data.test.labels, axis=1)
            accu_test = tf.reduce_mean(tf.cast(tf.equal(test_preds, true_labels), tf.float32))
            sigmoid_test_val = tf.reduce_max(sigmoid_test)
            softmax_test_val = tf.reduce_max(softmax_test)
            
            # 2. Random Noise
            noise_magnitude = 0.1
            random_noise = tf.random.uniform(
                mnist_data.test.images.shape,
                minval=-noise_magnitude,
                maxval=noise_magnitude
            )
            noisy_images = tf.clip_by_value(
                mnist_data.test.images + random_noise,
                0.0,
                1.0
            )
            softmax_noisy, sigmoid_noisy, _ = model.classifier(noisy_images)
            noisy_preds = tf.argmax(softmax_noisy, axis=1)
            accu_random = tf.reduce_mean(tf.cast(tf.equal(noisy_preds, true_labels), tf.float32))
            sigmoid_random = tf.reduce_max(sigmoid_noisy)
            softmax_random = tf.reduce_max(softmax_noisy)
            
            # 3. FGSM Examples
            perturbations = fast_gradient_method(
                model,
                mnist_data.test.images,
                **fgsm_params
            )
            adv_examples = tf.clip_by_value(
                mnist_data.test.images + perturbations,
                0.,
                1.
            )
            softmax_adv, sigmoid_adv, _ = model.classifier(adv_examples)
            adv_preds = tf.argmax(softmax_adv, axis=1)
            accu_adv = tf.reduce_mean(tf.cast(tf.equal(adv_preds, true_labels), tf.float32))
            sigmoid_adv_val = tf.reduce_max(sigmoid_adv)
            softmax_adv_val = tf.reduce_max(softmax_adv)
            
            # Print results
            print(f"{i}: epoch {i * batch_size // mnist_data.train.num_examples + 1}"
                  f" - test loss class: {c_loss:.4f} test loss gen: {g_loss:.4f}")
            print("Real test images     - Sigmoid: {:.4f}\tSoftmax: {:.4f}\taccuracy: {:.4f}".format(
                sigmoid_test_val, softmax_test_val, accu_test))
            print("Random noise images  - Sigmoid: {:.4f}\tSoftmax: {:.4f}\taccuracy: {:.4f}".format(
                sigmoid_random, softmax_random, accu_random))
            print("Adversarial examples - Sigmoid: {:.4f}\tSoftmax: {:.4f}\taccuracy: {:.4f}\n".format(
                sigmoid_adv_val, softmax_adv_val, accu_adv))
            
            # Save visualizations
            # 1. Generator outputs
            all_classes = tf.eye(10, dtype=tf.float32)
            samples, _ = model.generator(all_classes)
            fig = plot_generator(samples.numpy())
            plt.savefig(os.path.join(folder_out, f"gen_{i:06d}.png"), bbox_inches='tight')
            plt.close(fig)
            
            # 2. Hidden layer weights
            fig = plot_first_hidden(model.layer.weight.numpy())
            plt.savefig(os.path.join(folder_out, f"hidden_{i:06d}.png"), bbox_inches='tight')
            plt.close(fig)
            
            # 3. Adversarial examples
            fig = plot_generator(adv_examples[10:20].numpy())
            plt.savefig(os.path.join(folder_out, f"adv_{i:06d}.png"), bbox_inches='tight')
            plt.close(fig)
            
            # Store CSV rows
            accuracy_list.append([
                i,
                float(accu_test),
                float(accu_random),
                float(accu_adv),
                i // 500
            ])
            
            sigmoid_list.append([
                i,
                float(sigmoid_test_val),
                float(sigmoid_random),
                float(sigmoid_adv_val),
                i // 500
            ])
            
            softmax_list.append([
                i,
                float(softmax_test_val),
                float(softmax_random),
                float(softmax_adv_val),
                i // 500
            ])
            
            # TensorBoard logging
            with summary_writer.as_default():
                tf.summary.scalar("loss/classifier", c_loss, step=i)
                tf.summary.scalar("loss/generator", g_loss, step=i)
                tf.summary.scalar("accuracy/clean", accu_test, step=i)
                tf.summary.scalar("accuracy/noise", accu_random, step=i)
                tf.summary.scalar("accuracy/adversarial", accu_adv, step=i)
                tf.summary.scalar("max_output/sigmoid_test", sigmoid_test_val, step=i)
                tf.summary.scalar("max_output/softmax_test", softmax_test_val, step=i)
                tf.summary.scalar("max_output/sigmoid_noise", sigmoid_random, step=i)
                tf.summary.scalar("max_output/softmax_noise", softmax_random, step=i)
                tf.summary.scalar("max_output/sigmoid_adv", sigmoid_adv_val, step=i)
                tf.summary.scalar("max_output/softmax_adv", softmax_adv_val, step=i)

        # Add progress updates every 5000 steps
        if i % 5000 == 0:
            elapsed_hours = (time.time() - training_start_time) / 3600
            print(f"Training time so far: {elapsed_hours:.2f} hours")

    # After training loop, add timing summary
    training_time = time.time() - training_start_time
    total_time = time.time() - start_time
    
    # Print timing summary
    print("\n=== Timing Summary ===")
    print(f"Training time: {training_time/3600:.2f} hours ({training_time/60:.2f} minutes)")
    print(f"Total runtime: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
    
    # Save timing information
    timing_info = {
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        'training_time_hours': training_time/3600,
        'total_time_hours': total_time/3600,
        'config': config_name[config_num],
        'training_epochs': training_epochs,
        'batch_size': batch_size
    }
    
    # Save timing info to file
    timing_file = os.path.join(folder_csv, f"{config_name[config_num]}_training_metrics.txt")
    with open(timing_file, 'w') as f:
        for key, value in timing_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Timing information saved to: {timing_file}")

    # --------------------------------------------------------------------------------
    # Save final weights
    # --------------------------------------------------------------------------------
    weights_dir = folder_Weights  

    # Save final weights and biases
    np.save(os.path.join(weights_dir, "ANN_C_W1.npy"), model.layer.weight.numpy())
    if model.use_bias:
        np.save(os.path.join(weights_dir, "ANN_C_B1.npy"), model.layer.bias_forward.numpy())
    print(f"Saved final weights to: {weights_dir}")

    # --------------------------------------------------------------------------------
    # Save CSV logs
    # --------------------------------------------------------------------------------
    # Save accuracy, sigmoid, and softmax logs
    with open(os.path.join(folder_csv, "ANN_accuracy.csv"), "w", newline='') as f:
        csv.writer(f).writerows(accuracy_list)

    with open(os.path.join(folder_csv, "ANN_sigmoid.csv"), "w", newline='') as f:
        csv.writer(f).writerows(sigmoid_list)

    with open(os.path.join(folder_csv, "ANN_softmax.csv"), "w", newline='') as f:
        csv.writer(f).writerows(softmax_list)

    # --------------------------------------------------------------------------------
    # Save subsets for digits 2,3,4
    # --------------------------------------------------------------------------------
    # Test set subset
    test_labels_idx = np.argmax(mnist_data.test.labels, axis=1)
    mask_234 = np.isin(test_labels_idx, [2, 3, 4])

    X_test_234 = mnist_data.test.images[mask_234]
    y_test_234 = test_labels_idx[mask_234]
    X_test_234_flat = X_test_234.reshape((-1, 784))

    # Update subset saving location
    subset_path_X = os.path.join(folder_csv, "ANN_test_subset_234_images.npy")
    subset_path_y = os.path.join(folder_csv, "ANN_test_subset_234_labels.npy")
    np.save(subset_path_X, X_test_234_flat)
    np.save(subset_path_y, y_test_234)
    print("Saved test subset digits {2,3,4} to:", subset_path_X)

    # Train set subset
    train_labels_idx = np.argmax(mnist_data.train.labels, axis=1)
    mask_234_train = np.isin(train_labels_idx, [2, 3, 4])
    X_train_234 = mnist_data.train.images[mask_234_train]
    y_train_234 = train_labels_idx[mask_234_train]
    X_train_234_flat = X_train_234.reshape((-1, 784))

    train_subset_path_X = os.path.join(folder_csv, "ANN_train_subset_234_images.npy")
    train_subset_path_y = os.path.join(folder_csv, "ANN_train_subset_234_labels.npy")
    np.save(train_subset_path_X, X_train_234_flat)
    np.save(train_subset_path_y, y_train_234)
    print("Saved train subset digits {2,3,4} to:", train_subset_path_X)

    # Save noise and FGSM examples for digits 2,3,4
    # Generate final noise
    final_noise = tf.random.uniform(
        mnist_data.test.images.shape,
        minval=-0.1,
        maxval=0.1
    )
    test_noise = tf.clip_by_value(
        mnist_data.test.images + final_noise,
        0.0,
        1.0
    )

    # Generate final FGSM examples
    final_adv = mnist_data.test.images + fast_gradient_method(
        model,
        mnist_data.test.images,
        **fgsm_params
    )
    final_adv = tf.clip_by_value(final_adv, 0., 1.)

    # Save subsets
    X_noise_234 = test_noise[mask_234].numpy().reshape((-1, 784))
    X_fgsm_234 = final_adv[mask_234].numpy().reshape((-1, 784))

    noise_234_path = os.path.join(folder_csv, "ANN_test_subset_234_noise.npy")
    fgsm_234_path = os.path.join(folder_csv, "ANN_test_subset_234_fgsm.npy")
    np.save(noise_234_path, X_noise_234)
    np.save(fgsm_234_path, X_fgsm_234)

    print("Saved noise & FGSM for digits {2,3,4}:")
    print("  Noise =>", noise_234_path)
    print("  FGSM  =>", fgsm_234_path)

    # --------------------------------------------------------------------------------
    # Generate and save summary
    # --------------------------------------------------------------------------------
    def print_custom_summary(accuracy_list, sigmoid_list, softmax_list, output_file):
        acc_array = np.array(accuracy_list)
        sig_array = np.array(sigmoid_list)
        soft_array = np.array(softmax_list)
        batch_size = 100
        total_samples = mnist_data.train.num_examples
        
        with open(output_file, 'w') as f:
            # Best accuracy on test data
            best_test_idx = np.argmax(acc_array[:, 1])
            f.write("Best accuracy test data:\n")
            f.write(f"Index: {acc_array[best_test_idx, 4]} Iteration: {int(acc_array[best_test_idx, 0])} "
                    f"Epoch: {int(acc_array[best_test_idx, 0] * batch_size / total_samples)}\n")
            f.write(f"Test data: {acc_array[best_test_idx, 1]:.4f}\n")
            f.write(f"Test data with noise: {acc_array[best_test_idx, 2]:.4f}\n")
            f.write(f"Test data with FGSM: {acc_array[best_test_idx, 3]:.4f}\n")
            f.write(f"Sigmoid rate of maximum output of noise over real: {sig_array[best_test_idx, 2]/sig_array[best_test_idx, 1]:.4f}\n")
            f.write(f"Softmax rate of maximum output of noise over real: {soft_array[best_test_idx, 2]/soft_array[best_test_idx, 1]:.4f}\n\n")
            
            # Best accuracy on noisy test data
            best_noise_idx = np.argmax(acc_array[:, 2])
            f.write("Best accuracy test data with noise:\n")
            f.write(f"Index: {acc_array[best_noise_idx, 4]} Iteration: {int(acc_array[best_noise_idx, 0])} "
                    f"Epoch: {int(acc_array[best_noise_idx, 0] * batch_size / total_samples)}\n")
            f.write(f"Test data: {acc_array[best_noise_idx, 1]:.4f}\n")
            f.write(f"Test data with noise: {acc_array[best_noise_idx, 2]:.4f}\n")
            f.write(f"Test data with FGSM: {acc_array[best_noise_idx, 3]:.4f}\n")
            f.write(f"Sigmoid rate of maximum output of noise over real: {sig_array[best_noise_idx, 2]/sig_array[best_noise_idx, 1]:.4f}\n")
            f.write(f"Softmax rate of maximum output of noise over real: {soft_array[best_noise_idx, 2]/soft_array[best_noise_idx, 1]:.4f}\n\n")
            
            # Best accuracy on FGSM test data
            best_fgsm_idx = np.argmax(acc_array[:, 3])
            f.write("Best accuracy test data with FGSM:\n")
            f.write(f"Index: {acc_array[best_fgsm_idx, 4]} Iteration: {int(acc_array[best_fgsm_idx, 0])} "
                    f"Epoch: {int(acc_array[best_fgsm_idx, 0] * batch_size / total_samples)}\n")
            f.write(f"Test data: {acc_array[best_fgsm_idx, 1]:.4f}\n")
            f.write(f"Test data with noise: {acc_array[best_fgsm_idx, 2]:.4f}\n")
            f.write(f"Test data with FGSM: {acc_array[best_fgsm_idx, 3]:.4f}\n")
            f.write(f"Sigmoid rate of maximum output of noise over real: {sig_array[best_fgsm_idx, 2]/sig_array[best_fgsm_idx, 1]:.4f}\n")
            f.write(f"Softmax rate of maximum output of noise over real: {soft_array[best_fgsm_idx, 2]/soft_array[best_fgsm_idx, 1]:.4f}\n\n")
            
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

    # Generate and save summary
    summary_path = os.path.join(folder_csv, "ANN_summary.txt")
    print_custom_summary(
        accuracy_list,
        sigmoid_list,
        softmax_list,
        summary_path    
    )
    print(f"Saved summary to: {summary_path}")
    print("==== Finished Training ====")


    # After training is complete, before the final print statement:
    print("Generating training curves...")
    plot_training_curves(
        epoch_history,
        loss_history,
        accuracy_list,
        config_name[config_num],
        folder_csv
    )

#-----------------------------------------------------------------------------#
# Visualization Functions
#-----------------------------------------------------------------------------#
def plot_generator(samples):
    """Plot generated samples in a grid."""
    fig = plt.figure(figsize=(5, 2))
    gs = gridspec.GridSpec(2, 5)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')
    return fig

def plot_first_hidden(weights):
    """Plot weight matrix visualizations."""
    max_abs_val = max(abs(np.max(weights)), abs(np.min(weights)))
    fig = plt.figure(figsize=(5, 2))
    gs = gridspec.GridSpec(2, 5)
    gs.update(wspace=0.1, hspace=0.1)
    
    for i, weight in enumerate(np.transpose(weights)):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        im = plt.imshow(weight.reshape(28, 28), 
                       cmap="seismic_r",
                       vmin=-max_abs_val,
                       vmax=max_abs_val)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, ticks=[-max_abs_val, 0, max_abs_val])
    return fig

def plot_training_curves(epoch_history, loss_history, accuracy_list, config_name, folder_csv):
    """Plot and save training metrics."""
    # Convert accuracy list to numpy array for easier handling
    acc_array = np.array(accuracy_list)
    epochs = acc_array[:, 0] * 100 / 60000  # Convert iterations to epochs
    clean_acc = acc_array[:, 1]
    noise_acc = acc_array[:, 2]
    fgsm_acc = acc_array[:, 3]
    
    # Font sizes
    TITLE_SIZE = 24
    AXIS_LABEL_SIZE = 20
    TICK_LABEL_SIZE = 16
    LEGEND_SIZE = 18
    
    # Plot Loss vs. Epoch
    plt.figure(figsize=(12, 8))
    plt.plot(epoch_history, loss_history, 'b-', linewidth=2.5, label='Training Loss')
    plt.xlabel('Epoch', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Loss', fontsize=AXIS_LABEL_SIZE)
    plt.title('Loss vs. Epoch', fontsize=TITLE_SIZE, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.xticks(fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_csv, "loss_vs_epoch.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Accuracy vs. Epoch
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, clean_acc, 'g-', linewidth=2.5, label='Clean Test Data')
    plt.plot(epochs, noise_acc, 'b-', linewidth=2.5, label='Noisy Test Data')
    plt.plot(epochs, fgsm_acc, 'r-', linewidth=2.5, label='FGSM Test Data')
    plt.xlabel('Epoch', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Accuracy', fontsize=AXIS_LABEL_SIZE)
    plt.title('Accuracy vs. Epoch', fontsize=TITLE_SIZE, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.xticks(fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_csv, "accuracy_vs_epoch.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw data
    training_data = np.column_stack((epoch_history, loss_history))
    np.savetxt(
        os.path.join(folder_csv, "loss_vs_epoch.csv"), 
        training_data, 
        delimiter=',', 
        header='epoch,loss', 
        comments=''
    )
    
    accuracy_data = np.column_stack((epochs, clean_acc, noise_acc, fgsm_acc))
    np.savetxt(
        os.path.join(folder_csv, "accuracy_vs_epoch.csv"), 
        accuracy_data, 
        delimiter=',', 
        header='epoch,clean_accuracy,noise_accuracy,fgsm_accuracy', 
        comments=''
    )

if __name__ == "__main__":
    main()