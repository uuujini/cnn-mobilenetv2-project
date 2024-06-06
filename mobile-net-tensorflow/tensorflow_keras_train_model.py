import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(tf.__version__)
print("Available GPU Devices:", tf.config.list_physical_devices('GPU'))

def create_model(input_shape, num_classes):
    # Load MobileNetV2 without the top layer (head)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze layers

    # Create new top layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def main():
    img_width, img_height = 299, 299
    num_classes = 101
    batch_size = 20
    epochs = 10

    # Define paths
    train_data_dir = 'C:/Users/yujin/cnn-project/mobile-net/train'
    validation_data_dir = 'C:/Users/yujin/cnn-project/mobile-net/test'

    # Setup data generators
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width),
                                                        batch_size=batch_size, class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width),
                                                            batch_size=batch_size, class_mode='categorical')

    # Build the model
    model = create_model((img_height, img_width, 3), num_classes)

    # Compile the model
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # Setup callbacks
    checkpointer = ModelCheckpoint(filepath='best_model_mobilenetv2.h5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('training_mobilenetv2.log')

    # Train the model
    history = model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size,
                        validation_data=validation_generator, validation_steps=validation_generator.samples // batch_size,
                        epochs=epochs, verbose=1, callbacks=[checkpointer, csv_logger])

    # Save the final model
    model.save('final_model_mobilenetv2.h5')
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
