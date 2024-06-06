import os
import numpy as np
import tensorflow as tf
from tensorflow import data as tfdata
from tensorflow import image as tfimage
from tensorflow import io as tfio

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

def build_and_train_model():
    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(299, 299, 3), weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = base_model(inputs, training=False)
    x = tf.reduce_mean(x, axis=[1, 2])  # Global average pooling
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(101, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    train_data = load_data('C:/Users/yujin/cnn-project/mobile-net/train')
    validation_data = load_data('C:/Users/yujin/cnn-project/mobile-net/test')

    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.losses.SparseCategoricalCrossentropy()

    # Log file
    log_file_path = 'training_log.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch,Step,Loss,Accuracy,Validation Accuracy\n")

    # Training and validation
    for epoch in range(1):
        epoch_loss = []
        step_count = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_loss.append(loss.numpy())
            step_count += 1

            # Calculate accuracy for the batch
            predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
            batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_batch_train), tf.float32)).numpy()

            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy():.4f}, Accuracy: {batch_accuracy:.4f}")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{epoch+1},{step},{loss.numpy():.4f},{batch_accuracy:.4f}\n")

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

        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{epoch+1},{step},{avg_loss:.4f},{batch_accuracy:.4f},{val_accuracy:.4f}\n")

    # Save the model using the TensorFlow SavedModel format
    tf.saved_model.save(model, 'saved_model_mobilenetv2')
    print("Training completed and model saved.")

if __name__ == '__main__':
    build_and_train_model()
