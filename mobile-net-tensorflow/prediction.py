import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("TensorFlow version:", tf.__version__)
print("Available GPU Devices:", tf.config.list_physical_devices('GPU'))


# Creating a list of all the foods, in the argument I put the path to the folder that has all folders for food
def create_foodlist(path):
    list_ = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            list_.append(name)
    return list_


# Load the SavedModel format model
model_path = 'C:/Users/yujin/cnn-project/mobile-net-tensorflow/saved_model_mobilenetv2'
my_model = tf.saved_model.load(model_path)

# Display the signatures to find the correct output tensor name
print(my_model.signatures)

food_list = create_foodlist("food101/images")


# Function to help in predicting classes of new images loaded from my computer(for now)
def predict_class(model, images, show=True):
    for img_path in images:
        img = keras_image.load_img(img_path, target_size=(299, 299))
        img = keras_image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        # Perform prediction
        infer = model.signatures["serving_default"]
        pred = infer(tf.constant(img))
        output_tensor_name = list(pred.keys())[0]  # Get the output tensor name
        pred = pred[output_tensor_name]
        index = np.argmax(pred)  # Returns the indices of the maximum values along an axis
        food_list.sort()
        pred_value = food_list[index]

        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(pred_value)
            plt.show()


# Add the images you want to predict into a list (these are in the WD)
images = []
images.append('sc.jpg')
images.append('vb1.jpg')
images.append('vp.jpg')
images.append('es.jpg')

print("PREDICTIONS BASED ON PICTURES UPLOADED")
predict_class(my_model, images, True)
