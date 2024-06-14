import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import easygui
from rembg import remove

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='model webcam.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_path = easygui.fileopenbox(title='Select Image File')
# img = image.load_img(test_image_path)
img = cv2.imread(str(input_path))
img = remove(img)
img.save('ok.png')