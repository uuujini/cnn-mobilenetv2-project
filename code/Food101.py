import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

# 새로운 MobileNetV2 모델을 불러오는 대신에 직접 구현한 MobileNetV2 함수를 임포트합니다.
from MobileNetV2 import MobileNetV2

n_classes = 101
img_width, img_height = 299, 299
train_data_dir = 'C:/Users/SAMSUNG/OneDrive/바탕 화면/DS/archive/train'
validation_data_dir = 'C:/Users/SAMSUNG/OneDrive/바탕 화면/DS/archive/test'
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
# 기존 MobileNetV2 코드를 대체하기 위해 새로운 MobileNetV2 함수를 호출합니다.
mbv2 = MobileNetV2(input_shape=(299, 299, 3), num_classes=101)

# 기존 모델의 출력 레이어를 가져옵니다.
x = mbv2.output

# Flatten 레이어를 추가하여 4차원을 2차원으로 변환합니다.
x = Flatten()(x)

# 새로운 Dense 레이어를 추가합니다.
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

# 출력 레이어를 추가합니다.
predictions = Dense(101, activation='softmax')(x)

# 모델을 정의합니다.
model = Model(inputs=mbv2.input, outputs=predictions)

# 모델을 컴파일합니다.
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 데이터를 준비합니다.
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

# 모델을 훈련합니다.
checkpointer = ModelCheckpoint(filepath='best_model_3class_sept.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=1,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

# 모델을 저장합니다.
model.save('model_trained.h5')