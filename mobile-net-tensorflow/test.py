import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    """ 이미지 파일을 읽고 전처리하는 함수 """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [299, 299])  # 모델 입력 크기에 맞게 조정
    image = image / 255.0  # 이미지를 [0, 1] 범위로 정규화
    return image

def load_model(model_path):
    """ 저장된 TensorFlow 모델을 로드하는 함수 """
    model = tf.saved_model.load(model_path)
    return model

def predict_image(model, image_path):
    """ 이미지에 대한 예측을 수행하는 함수 """
    image = load_and_preprocess_image(image_path)
    image_np = np.expand_dims(image, axis=0)  # 배치 차원 추가

    infer = model.signatures['serving_default']
    prediction = infer(tf.constant(image_np))

    predicted_index = np.argmax(prediction['output_0'].numpy(), axis=1)[0]  # 가장 높은 확률의 인덱스
    return predicted_index, image

# 모델 경로 설정
model_path = 'trained_model_mobilenetv2_custom'

# 모델 로드
model = load_model(model_path)

# 예측할 이미지 경로 설정
image_path = 'C:/Users/yujin/cnn-project/mobile-net/test/bibimbap/2556.jpg'

# 이미지 예측
predicted_index, image = predict_image(model, image_path)

# 클래스 리스트 - 예측 인덱스에 따른 클래스 이름을 출력할 수 있도록 수정하세요.
class_names = ['pizza', 'bibimbap', ...]  # 클래스 이름 리스트를 적절히 수정하세요.

# 예측 결과 출력
print("Predicted class index:", predicted_index)
print("Predicted class name:", class_names[predicted_index])

# 이미지 출력
plt.imshow(image)
plt.title(f'Predicted: {class_names[predicted_index]}')
plt.axis('off')
plt.show()
