"""
@author: Robert Kamunde
"""
from keras import utils
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model

#creating a list of all the foods, in the argument i put the path to the folder that has all folders for food
def create_foodlist(path):
    list_ = list()
    for root, dirs, files in os.walk(path, topdown=False):
      for name in dirs:
        list_.append(name)
    return list_

#loading the model i trained and finetuned
my_model = load_model('trained_MobileNetV2_KerasAPI.h5', compile=False)
food_list = create_foodlist("food101/images")

#function to help in predicting classes of new images loaded from my computer(for now)
def predict_class(model, images, show = True):
  for img in images:
    img = utils.load_img(img, grayscale=False, color_mode='rgb', target_size=(299, 299))

    img = utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    pred = model.predict(img)
    index = np.argmax(pred)    #Returns the indices of the maximum values along an axis, In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
    food_list.sort()
    pred_value = food_list[index]
    if show:
        plt.imshow(img[0])
        plt.axis('off')
        plt.title(pred_value)
        plt.show()

#add the images you want to predict into a list (these are in the WD)
images = []
images.append('val_images/cc.jpg')
images.append('val_images/cr.jpg')
images.append('val_images/ss.jpg')
images.append('val_images/ty.jpg')


print("PREDICTIONS BASED ON PICTURES UPLOADED")
predict_class(my_model, images, True)
