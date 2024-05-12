# 사전학습 모델 불러오기 : 케라스에서 클래스 형태로 제공함
from tensorflow.keras.applications.vgg16 import VGG16

# weight, include_top 파라미터 설정
model = VGG16(weights='imagenet', include_top=True)
model.summary()

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np

from google.colab import drive

drive.mount('/content/gdrive')

cd / content / gdrive / MyDrive / CDSS
딥러닝
강의 / 심화_1

img = Image.open('들3.jpg')
img.size
plt.imshow(np.asarray(img))

w, h = img.size
s = min(w, h)
y = (h - s) // 2
x = (w - s) // 2

print(w, h, x, y, s)
img = img.crop((x, y, x + s, y + s))
# 4-tuple defining the left, upper, right, and lower pixel coordinate
plt.imshow(np.asarray(img))
img.size

# VGG16이 입력받는 이미지크기 확인
model.layers[0].input_shape

# 이미지 리사이즈
target_size = 224
img = img.resize((target_size, target_size))  # resize from 280x280 to 224x224
plt.imshow(np.asarray(img))

img.size  # 변경된 크기 확인

# numpy array로 변경
np_img = image.img_to_array(img)
np_img.shape  # (224, 224, 3)

# 4차원으로 변경
img_batch = np.expand_dims(np_img, axis=0)
img_batch.shape