import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# 모델 로드
model_path = 'trained_model_mobilenetv2_custom'
model = tf.keras.models.load_model(model_path)

# 데이터셋 로드 및 전처리
test_data_path = 'C:/Users/yujin/cnn-project/mobile-net/test'  # 테스트 데이터셋 폴더 경로
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # MobileNetV2 입력 크기에 맞춤

test_dataset = image_dataset_from_directory(
    test_data_path,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# 모델 평가
loss, accuracy = model.evaluate(test_dataset)

# 결과 그래프로 표시
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy])
plt.title('Model Performance on Test Dataset')
plt.ylabel('Accuracy')
plt.show()
