import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

def create_foodlist(path):
    list_ = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            list_.append(name)
    return list_

# Load the model from the SavedModel format
my_model = tf.saved_model.load('C:/Users/yujin/cnn-project/mobile-net-tensorflow/saved_model_mobilenetv2_basemodel')

food_list = create_foodlist("food101/images")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Print available keys in the model's signature
infer = my_model.signatures["serving_default"]
print("Model signature keys:", infer.structured_outputs.keys())

def predict_class(model, images, show=True):
    infer = model.signatures["serving_default"]
    for img_path in images:
        img = preprocess_image(img_path)
        pred = infer(tf.constant(img))
        # Print available keys in prediction result
        print("Prediction result keys:", pred.keys())
        # Use the correct key for the prediction
        key = list(pred.keys())[0]
        index = np.argmax(pred[key])
        food_list.sort()
        pred_value = food_list[index]
        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(pred_value)
            plt.show()

images = [
    'cr.jpg',
    'ty.jpg',
    'vp.jpg',
    'es.jpg'
]

print("PREDICTIONS BASED ON PICTURES UPLOADED")
predict_class(my_model, images, True)
