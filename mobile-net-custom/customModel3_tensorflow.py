import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_mobilenet_v2(input_shape=(299, 299, 3), num_classes=101):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model

def train_mobilenet():
    img_width, img_height = 299, 299
    train_data_dir = 'C:/Users/yujin/cnn-project/mobile-net-custom/train'
    validation_data_dir = 'C:/Users/yujin/cnn-project/mobile-net-custom/test'
    nb_train_samples = 75750
    nb_validation_samples = 25250
    batch_size = 20

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    model = create_mobilenet_v2()

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='best_model_3class_sept.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('history.log')

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=nb_train_samples // batch_size,
                                  validation_data=validation_generator,
                                  validation_steps=nb_validation_samples // batch_size,
                                  epochs=1, # EPOCHS 우선 1로 지정
                                  verbose=1,
                                  callbacks=[csv_logger, checkpointer])

    model.save('custom_model_trained.h5')

train_mobilenet()
