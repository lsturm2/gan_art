# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
#tf.enable_eager_execution()

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import json
from IPython.display import clear_output
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

DATABASE_PATH = "./data/image_list.txt"
TEST_DIR = "../wikiart/test"

def main():
	datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

	training_set = datagen.flow_from_directory(
		TEST_DIR,
		target_size=(256, 256),
		color_mode="rgb",
		batch_size=1,
		class_mode=None,
		shuffle=False, # maybe later
		seed=42)
	

def get_image_paths():
	image_paths = []
	with open(DATABASE_PATH) as file:
		image_paths = json.load(file)

	image_paths = image_paths[:3]
	return image_paths





if __name__ == '__main__':
	main()
