import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.image import pad_to_bounding_box
from tensorflow.image import central_crop
from tensorflow.image import resize

# 이미지 불러오기
bgd = image.load_img('C:/Users/Yeong/Desktop/CDSS강의_심화/심화_강의자료1/python_code/들3.jpg')
bgd_vector = np.asarray(image.img_to_array(bgd))
bgd_vector = bgd_vector / 255

# 이미지 형태 확인
bgd_vector.shape

# 이미지 확인
plt.imshow(bgd_vector)
plt.show()

# 이미지의 변경할 크기 설정
target_height = 4500
target_width = 4500

# 현재 이미지의 크기 지정
source_height = bgd_vector.shape[0]
source_width = bgd_vector.shape[1]

# padding 실시 : pad_to_bounding_box 사용
bgd_vector_pad = pad_to_bounding_box(bgd_vector,
                                     int((target_height - source_height) / 2),
                                     int((target_width - source_width) / 2),
                                     target_height,
                                     target_width)

# 이미지 형태 확인
bgd_vector_pad.shape

# 이미지 확인
plt.imshow(bgd_vector_pad)
plt.show()

# 이미지 저장
image.save_img(r'C:\Users\Yeong\Desktop\CDSS강의_심화\심화_강의자료1\python_code\cat1_pad.png', cat_vector_pad)

# 가운데를 중심으로 50%만 crop
bgd_vector_crop = central_crop(bgd_vector, .5)

bgd_vector_crop.shape

plt.imshow(bgd_vector_crop)
plt.show()

w, h = bgd.size

s = min(w, h)  # 둘 중에 작은 것 기준으로 자름
y = (h - s) // 2
x = (w - s) // 2

print(w, h, x, y, s)

# 좌, 위, 오른쪽, 아래 픽셀 설정
bgd = bgd.crop((x, y, x + s, y + s))
plt.imshow(np.asarray(bgd))
bgd.size

bgd_vector_resize = resize(bgd_vector, (300, 300))

bgd_vector_resize.shape

plt.imshow(bgd_vector_resize)