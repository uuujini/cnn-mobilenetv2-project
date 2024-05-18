import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import MobileNetV2

# 데이터셋 경로 설정
train_data_dir = 'C:/Users/yujin/cnn-project/mobile-net-custom/train'
validation_data_dir = 'C:/Users/yujin/cnn-project/mobile-net-custom/test'

# 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# MobileNetV2 모델 로드 (include_top=False로 최상위 레이어 제외)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 중간 레이어를 수정하고 추가 레이어를 연결
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

# 커스텀 레이어 추가 예제 (필요한 경우 수정 가능)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

# 최종 출력 레이어
predictions = Dense(101, activation='softmax')(x)

# 전체 모델 구성
model = Model(inputs=base_model.input, outputs=predictions)

# 일부 레이어를 학습 불가능하도록 고정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
checkpointer = ModelCheckpoint(filepath='best_model.h5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training_log.csv')

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10, # 필요한 경우 에포크 수 조정
    callbacks=[checkpointer, csv_logger]
)

# 모델 저장
model.save('custom_mobilenet_v2.h5')
