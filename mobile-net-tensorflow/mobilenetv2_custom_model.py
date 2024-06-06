# Final tensorflow mobilenetv2 custom model

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


def conv_block(inputs, filters, kernel_size, strides=1):
    conv = tf.nn.conv2d(inputs, filters, strides=[1, strides, strides, 1], padding='SAME')
    conv = tf.nn.relu(conv)
    conv = tf.nn.batch_normalization(conv, mean=0, variance=1, offset=0, scale=1, variance_epsilon=0.001)
    return conv


def depthwise_conv_block(inputs, depthwise_filter, pointwise_filter, strides=1):
    depthwise_conv = tf.nn.depthwise_conv2d(inputs, depthwise_filter, strides=[1, strides, strides, 1], padding='SAME')
    depthwise_conv = tf.nn.relu(depthwise_conv)
    depthwise_conv = tf.nn.batch_normalization(depthwise_conv, mean=0, variance=1, offset=0, scale=1,
                                               variance_epsilon=0.001)
    pointwise_conv = tf.nn.conv2d(depthwise_conv, pointwise_filter, strides=[1, 1, 1, 1], padding='SAME')
    pointwise_conv = tf.nn.relu(pointwise_conv)
    pointwise_conv = tf.nn.batch_normalization(pointwise_conv, mean=0, variance=1, offset=0, scale=1,
                                               variance_epsilon=0.001)
    return pointwise_conv


def build_custom_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial convolution block
    filters = tf.Variable(tf.random.normal([3, 3, 3, 32]))
    x = conv_block(inputs, filters, kernel_size=3, strides=2)

    # Depthwise separable convolution blocks
    depthwise_filter = tf.Variable(tf.random.normal([3, 3, 32, 1]))
    pointwise_filter = tf.Variable(tf.random.normal([1, 1, 32, 64]))
    x = depthwise_conv_block(x, depthwise_filter, pointwise_filter, strides=1)

    depthwise_filter = tf.Variable(tf.random.normal([3, 3, 64, 1]))
    pointwise_filter = tf.Variable(tf.random.normal([1, 1, 64, 128]))
    x = depthwise_conv_block(x, depthwise_filter, pointwise_filter, strides=2)

    depthwise_filter = tf.Variable(tf.random.normal([3, 3, 128, 1]))
    pointwise_filter = tf.Variable(tf.random.normal([1, 1, 128, 256]))
    x = depthwise_conv_block(x, depthwise_filter, pointwise_filter, strides=2)

    depthwise_filter = tf.Variable(tf.random.normal([3, 3, 256, 1]))
    pointwise_filter = tf.Variable(tf.random.normal([1, 1, 256, 512]))
    x = depthwise_conv_block(x, depthwise_filter, pointwise_filter, strides=2)

    # Global average pooling
    x = tf.reduce_mean(x, axis=[1, 2])

    # Dense layers
    dense_weights = tf.Variable(tf.random.normal([512, 128]))
    dense_bias = tf.Variable(tf.zeros([128]))
    x = tf.nn.relu(tf.matmul(x, dense_weights) + dense_bias)

    dense_weights = tf.Variable(tf.random.normal([128, num_classes]))
    dense_bias = tf.Variable(tf.zeros([num_classes]))
    outputs = tf.nn.softmax(tf.matmul(x, dense_weights) + dense_bias)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def build_and_train_model():
    model = build_custom_model((299, 299, 3), 101)
    train_data = load_data('C:/Users/yujin/cnn-project/mobile-net/train')
    validation_data = load_data('C:/Users/yujin/cnn-project/mobile-net/test')

    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.losses.SparseCategoricalCrossentropy()

    # Log file
    log_file_path = 'training_log.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch,Step,Loss,Accuracy,Validation Accuracy\n")

    total_steps = sum(1 for _ in train_data) * 30  # Total steps for all epochs
    completed_steps = 0

    # Training and validation
    for epoch in range(30):  # Increase number of epochs for more training
        print(f"Starting epoch {epoch + 1}")
        epoch_loss = []
        step_count = 0
        epoch_start_time = time.time()

        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            step_start_time = time.time()

            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_loss.append(loss.numpy())
            step_count += 1
            completed_steps += 1

            # Calculate accuracy for the batch
            predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
            batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_batch_train), tf.float32)).numpy()

            step_time = time.time() - step_start_time
            time_per_step = step_time
            remaining_time = (total_steps - completed_steps) * time_per_step
            remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(remaining_time))

            if step % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, Step {step}, Loss: {loss.numpy():.4f}, Accuracy: {batch_accuracy:.4f}, Remaining Time: {remaining_time_str}")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{epoch + 1},{step},{loss.numpy():.4f},{batch_accuracy:.4f}\n")

        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch + 1}/{30}, Average Loss: {avg_loss:.4f}")

        # Validation
        val_accuracy_metric = tf.metrics.SparseCategoricalAccuracy()
        for x_batch_val, y_batch_val in validation_data:
            val_logits = model(x_batch_val, training=False)
            val_accuracy_metric.update_state(y_batch_val, val_logits)
        val_accuracy = val_accuracy_metric.result()
        val_accuracy_metric.reset_states()
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.4f}")

        with open(log_file_path, 'a') as log_file:
            log_file.write(
                f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n")

    # Save the model using the TensorFlow SavedModel format
    tf.saved_model.save(model, 'saved_model_mobilenetv2')
    print("Training completed and model saved.")


if __name__ == '__main__':
    build_and_train_model()
