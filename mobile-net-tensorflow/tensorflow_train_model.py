import os
import numpy as np
import tensorflow as tf
from tensorflow import data as tfdata
from tensorflow import image as tfimage
from tensorflow import io as tfio
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("TensorFlow version:", tf.__version__)
print("Available GPU Devices:", tf.config.list_physical_devices('GPU'))

def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [299, 299])
    img = img / 255.0  # Normalize to [0, 1]
    return img

def load_data(data_dir):
    categories = os.listdir(data_dir)
    image_paths = []
    labels = []
    for idx, category in enumerate(categories):
        cat_path = os.path.join(data_dir, category)
        img_paths = [os.path.join(cat_path, name) for name in os.listdir(cat_path)]
        image_paths.extend(img_paths)
        labels.extend([idx] * len(img_paths))

    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels)
    dataset = tfdata.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def dense_layer(input_tensor, units, activation=None):
    # Weights initialization
    input_dim = int(input_tensor.shape[-1])
    weights = tf.Variable(tf.random.normal([input_dim, units]))
    bias = tf.Variable(tf.zeros([units]))
    # Fully connected layer implementation
    x = tf.matmul(input_tensor, weights)
    x = tf.nn.bias_add(x, bias)
    if activation is not None:
        x = activation(x)
    return x

def build_and_train_model():
    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(299, 299, 3), weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = base_model(inputs, training=False)
    x = tf.reduce_mean(x, axis=[1, 2])  # Global average pooling

    # Using the custom dense_layer function
    x = dense_layer(x, 128, activation=tf.nn.relu)
    x = tf.nn.dropout(x, rate=0.2)
    outputs = dense_layer(x, 101, activation=tf.nn.softmax)

    model = tf.keras.Model(inputs, outputs)

    train_data = load_data('C:/Users/yujin/cnn-project/mobile-net/train')
    validation_data = load_data('C:/Users/yujin/cnn-project/mobile-net/test')

    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.losses.SparseCategoricalCrossentropy()

    log_file = open("training_log.txt", "w")

    # Training and validation
    for epoch in range(10):
        start_time = time.time()
        epoch_loss = []
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_loss.append(loss.numpy())

            if step % 10 == 0:
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time * (len(train_data) - step - 1) / (step + 1)
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy():.4f}, Elapsed time: {elapsed_time:.2f}s, Remaining time: {remaining_time:.2f}s")

        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch+1}/{10}, Average Loss: {avg_loss:.4f}")

        # Validation
        val_accuracy_metric = tf.metrics.SparseCategoricalAccuracy()
        for x_batch_val, y_batch_val in validation_data:
            val_logits = model(x_batch_val, training=False)
            val_accuracy_metric.update_state(y_batch_val, val_logits)
        val_accuracy = val_accuracy_metric.result()
        val_accuracy_metric.reset_states()
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}")

        log_file.write(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n")

    log_file.close()

    # Save the model
    model.save('final_model_mobilenetv2.h5')
    print("Training completed and model saved.")

if __name__ == '__main__':
    print("Starting training...")
    build_and_train_model()
