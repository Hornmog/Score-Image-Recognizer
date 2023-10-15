import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def pre_process_bit(img):
    img = cv2.resize(img, (28, 28))
    img = np.invert(img)
    ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, (3, 3))
    return img




new_model = tf.keras.models.load_model('handwritten.model')

imageNumber = 3
imagePath = f"TestImages/digits/{imageNumber}.png"

# import images
try:
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    image = pre_process_bit(image)
    prediction = new_model.predict(np.array([image]))
    print(f"Prediction: {np.argmax(prediction)}")
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()


except Exception as e:
    print(e)


 # function to process image

