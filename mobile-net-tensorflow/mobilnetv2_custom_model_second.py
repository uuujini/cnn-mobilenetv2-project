import os
import numpy as np
import tensorflow as tf
from tensorflow import data as tfdata
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("TensorFlow version:", tf.__version__)
print("Available GPU Devices:", tf.config.list_physical_devices('GPU'))

class CustomBatchNorm(tf.Module):
    def __init__(self, epsilon=0.001):
        super().__init__()
        self.epsilon = epsilon
        self.scale = tf.Variable(tf.ones([]), trainable=True)
        self.offset = tf.Variable(tf.zeros([]), trainable=True)

    def __call__(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)
        return tf.nn.batch_normalization(inputs, mean, variance, self.offset, self.scale, self.epsilon)

class ConvBlock(tf.Module):
    def __init__(self, filters, kernel_size, strides=1, use_dropout=False):
        super().__init__()
        self.strides = strides
        self.use_dropout = use_dropout
        self.conv_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=[kernel_size, kernel_size, 3, filters]))
        self.bn = CustomBatchNorm()
        if self.use_dropout:
            self.dropout = tf.Variable(tf.constant(0.2), trainable=False)

    def __call__(self, inputs):
        conv = tf.nn.conv2d(inputs, self.conv_weights, strides=[1, self.strides, self.strides, 1], padding='SAME')
        conv = tf.nn.relu(conv)
        conv = self.bn(conv)
        if self.use_dropout:
            conv = tf.nn.dropout(conv, rate=self.dropout)
        return conv

class DepthwiseConvBlock(tf.Module):
    def __init__(self, input_channels, depth_multiplier, pointwise_filters, strides=1, use_dropout=False):
        super().__init__()
        self.strides = strides
        self.use_dropout = use_dropout
        self.depthwise_filter = tf.Variable(tf.initializers.GlorotUniform()(shape=[3, 3, input_channels, depth_multiplier]))
        self.pointwise_filter = tf.Variable(tf.initializers.GlorotUniform()(shape=[1, 1, input_channels * depth_multiplier, pointwise_filters]))
        self.bn1 = CustomBatchNorm()
        self.bn2 = CustomBatchNorm()
        if self.use_dropout:
            self.dropout = tf.Variable(tf.constant(0.2), trainable=False)

    def __call__(self, inputs):
        depthwise_conv = tf.nn.depthwise_conv2d(inputs, self.depthwise_filter, strides=[1, self.strides, self.strides, 1], padding='SAME')
        depthwise_conv = tf.nn.relu(depthwise_conv)
        depthwise_conv = self.bn1(depthwise_conv)

        pointwise_conv = tf.nn.conv2d(depthwise_conv, self.pointwise_filter, strides=[1, 1, 1, 1], padding='SAME')
        pointwise_conv = tf.nn.relu(pointwise_conv)
        pointwise_conv = self.bn2(pointwise_conv)

        if self.use_dropout:
            pointwise_conv = tf.nn.dropout(pointwise_conv, rate=self.dropout)
        return pointwise_conv

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

import tensorflow as tf

class CustomModel(tf.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Convolutional block 1
        self.conv_block1 = ConvBlock(32, 3, strides=2, use_dropout=True)
        # Depthwise separable convolution block 1
        self.depthwise_conv_block1 = DepthwiseConvBlock(32, 1, 64, strides=1, use_dropout=True)
        # Depthwise separable convolution block 2
        self.depthwise_conv_block2 = DepthwiseConvBlock(64, 1, 128, strides=2, use_dropout=True)
        # Depthwise separable convolution block 3
        self.depthwise_conv_block3 = DepthwiseConvBlock(128, 1, 256, strides=2, use_dropout=True)
        # Depthwise separable convolution block 4
        self.depthwise_conv_block4 = DepthwiseConvBlock(256, 1, 512, strides=2, use_dropout=True)
        # Dense layer setup
        self.dense_weights1 = tf.Variable(tf.initializers.GlorotUniform()(shape=[512, 128]))
        self.dense_bias1 = tf.Variable(tf.zeros([128]))
        self.dense_weights2 = tf.Variable(tf.initializers.GlorotUniform()(shape=[128, num_classes]))
        self.dense_bias2 = tf.Variable(tf.zeros([num_classes]))

    def __call__(self, inputs, training=False):
        x = self.conv_block1(inputs)
        x = self.depthwise_conv_block1(x)
        x = self.depthwise_conv_block2(x)
        x = self.depthwise_conv_block3(x)
        x = self.depthwise_conv_block4(x)
        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling
        x = tf.nn.relu(tf.matmul(x, self.dense_weights1) + self.dense_bias1)
        outputs = tf.nn.softmax(tf.matmul(x, self.dense_weights2) + self.dense_bias2)
        return outputs

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 299, 299, 3], dtype=tf.float32)])
    def serving_default(self, inputs):
        return self.__call__(inputs, training=False)

def plot_accuracy(log_file_path):
    epochs, train_accuracies, val_accuracies = [], [], []
    with open(log_file_path, 'r') as file:
        next(file)  # 헤더 스킵
        for line in file:
            data = line.strip().split(',')
            epoch, train_accuracy, val_accuracy = int(data[0]), float(data[3]), float(data[5])
            epochs.append(epoch)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def build_and_train_model():
    model = CustomModel((299, 299, 3), 101)
    train_data = load_data('C:/Users/yujin/cnn-project/mobile-net/train')
    validation_data = load_data('C:/Users/yujin/cnn-project/mobile-net/test')

    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.losses.SparseCategoricalCrossentropy()

    log_file_path = 'custom_training_log.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch,Step,Loss,Accuracy,Val_Loss,Val_Accuracy\n")

    total_steps = sum(1 for _ in train_data) * 30
    completed_steps = 0

    for epoch in range(1):  # Increase number of epochs for more training
        print(f"Starting epoch {epoch + 1}")
        epoch_loss = []
        epoch_accuracy = []
        step_count = 0

        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(loss.numpy())

            # Calculate accuracy for the batch
            predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
            batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_batch_train), tf.float32)).numpy()
            epoch_accuracy.append(batch_accuracy)
            step_count += 1
            completed_steps += 1

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.numpy():.4f}, Accuracy: {batch_accuracy:.4f}")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{epoch + 1},{step},{loss.numpy():.4f},{batch_accuracy:.4f},,\n")

        avg_loss = np.mean(epoch_loss)
        avg_accuracy = np.mean(epoch_accuracy)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

        # Validation
        val_loss = 0
        val_accuracy_metric = tf.metrics.SparseCategoricalAccuracy()
        for x_batch_val, y_batch_val in validation_data:
            val_logits = model(x_batch_val, training=False)
            val_loss_batch = loss_fn(y_batch_val, val_logits)
            val_loss += val_loss_batch.numpy()
            val_accuracy_metric.update_state(y_batch_val, val_logits)
        val_accuracy = val_accuracy_metric.result().numpy()
        val_loss /= len(validation_data)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{epoch + 1},,,,{val_loss:.4f},{val_accuracy:.4f}\n")

    tf.saved_model.save(model, 'trained_model_mobilenetv2_custom', signatures={'serving_default': model.serving_default})
    print("Training completed and model saved.")

if __name__ == '__main__':
    build_and_train_model()
    plot_accuracy('custom_training_log.txt')
