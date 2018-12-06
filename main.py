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
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop


DATABASE_PATH = "./data/image_list.txt"
TEST_DIR = "../wikiart/train"

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
		batch_size=16,
		class_mode=None,
		shuffle=True, # maybe now?
		seed=42)

	gen = generator()
	disc = discriminator_model()
	adv = adversarial_model()

	train(training_set, gen, disc, adv)
	
	# print(next(training_set).shape)
	# print(next(training_set).shape)
	# print(next(training_set))


def train(x_train, generator, discriminator, adversarial, train_steps=2, batch_size=16):
	save_int = 10
	for i in range(len(x_train)):
		images_train = next(x_train)
		noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
		images_fake = generator.predict(noise)
		print("images_train shape:", images_train.shape)
		print("shape of images_fake:", images_fake.shape)
		x = np.concatenate((images_train, images_fake))
		y = np.ones([2*batch_size, 3])
		y[batch_size:, :] = 0
		d_loss = discriminator.train_on_batch(x, y)

		y = np.ones([batch_size, 3])
		noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
		a_loss = adversarial.train_on_batch(noise, y)
		log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
		print(log_mesg)
		log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
		print(log_mesg)
		

		if i % save_int == 0:
			display_images(generator, images_train, save2file=True, samples=noise.shape[0],\
					noise=noise, step=(i+1))


def display_images(generator, images_train, save2file=False, fake=True, samples=3, noise=None, step=0):
	filename = "wahoo_%d.png" % step
	images = generator.predict(noise)

	plt.figure(figsize=(10, 10))
	image = images[0]
	image = np.reshape(image, [256, 256, 3]) * 255
	print(image)
	plt.imshow(image)
	plt.axis('off')
	plt.show()

def adversarial_model():
	optimizer = RMSprop(lr=0.0001, decay=3e-8)
	model = Sequential()
	model.add(generator())
	model.add(discriminator())
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


def discriminator_model():
	optimizer = RMSprop(lr=0.0002, decay=6e-8)
	model = Sequential()
	model.add(discriminator())
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


def discriminator():
	model = Sequential()
	dropout = 0.5 # between 0.4 - 0.7
	input_shape = (256, 256, 3)
	
	model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=input_shape, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(dropout))

	model.add(Flatten())
	model.add(Dense(3))
	model.add(Activation('sigmoid'))
	model.summary()

	return model


def generator():
	model = Sequential()
	
	dropout = 0.4
	depth = 64 + 64 + 64
	dim = 64

	model.add(Dense(dim * dim * depth, input_dim=100))
	model.add(BatchNormalization(momentum=0.9))
	model.add(Activation('relu'))
	model.add(Reshape((dim, dim, depth)))
	model.add(Dropout(dropout))

	model.add(UpSampling2D())
	model.add(Conv2DTranspose(int(depth / 2), kernel_size=3, padding='same'))
	model.add(BatchNormalization(momentum=0.9))
	model.add(Activation('relu'))

	model.add(UpSampling2D())
	model.add(Conv2DTranspose(int(depth / 4), kernel_size=3, padding='same'))
	model.add(Activation('relu'))

	model.add(Conv2DTranspose(3, kernel_size=3, padding='same'))
	model.add(Activation('tanh'))
	model.summary()

	return model


def get_image_paths():
	image_paths = []
	with open(DATABASE_PATH) as file:
		image_paths = json.load(file)

	image_paths = image_paths[:3]
	return image_paths





if __name__ == '__main__':
	main()
