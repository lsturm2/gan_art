# This file does basic analystics (counting) of the dataset 
# and also stores these objects as json objects that can be
# used later. It creates two maps: linking styles to images
# and images to styles. It also creates a list of paths to
# images.

import os
import json
DATABASE_DIR = "/Users/lukesturm/fall/CS238/wikiart"
IMAGE_TO_STYLE_PATH = "data/image_to_style.txt"
STYLE_TO_IMAGE_PATH = "data/style_to_image.txt"
IMAGE_LIST_PATH = "data/image_list.txt"

def main():
	image_to_style = create_image_to_style()
	style_to_image = create_style_to_image(image_to_style)
	image_path_list = create_image_path_list()
	save_data(image_to_style, style_to_image, image_path_list)
	print_image_analytics(style_to_image, image_path_list)


def save_data(image_to_style, style_to_image, image_path_list):
	write_to_file(IMAGE_TO_STYLE_PATH, image_to_style)
	write_to_file(STYLE_TO_IMAGE_PATH, style_to_image)
	write_to_file(IMAGE_LIST_PATH, image_path_list)


def write_to_file(path, data_struct):
	file = open(path, 'w')
	file.write(json.dumps(data_struct, sort_keys=True, indent=4))
	file.close()

# Function: print_image_analytics
# -------------------------------
# This function prints the number of total photos 
# and the number of photos by category
def print_image_analytics(style_to_image, image_path_list):
	print("Total number of images: %d" % (len(image_path_list)))
	for style in style_to_image:
		print("Number of pictures for %s: %d" % (style, len(style_to_image[style])))


# Function: create_image_path_list
# --------------------------------
# This function creates a list of full image paths
# based on the DATABASE_DIR. Update that variable
# for this to work.
def create_image_path_list():
	image_list = []
	style_dirs = os.listdir(DATABASE_DIR)
	for style in style_dirs:
		style_path = "%s/%s" % (DATABASE_DIR, style)
		images = os.listdir(style_path)
		for image in images:
			image_path = "%s/%s" % (style_path, image)
			image_list.append(image_path)

	return image_list


# Function: create_style_to_image
# -------------------------------
# Creates a map from styles to list of image names
def create_style_to_image(image_to_style):
	style_to_image = {}

	for image in image_to_style:

		style = image_to_style[image]
		if style not in style_to_image:
			style_to_image[style] = []

		style_to_image[style].append(image)

	return style_to_image


# Function: create_image_to_style
# -------------------------------
# Creates a map from images to styles
def create_image_to_style():
	image_to_style = {}
	style_dirs = os.listdir(DATABASE_DIR)

	for style in style_dirs:
		style_path = "%s/%s" % (DATABASE_DIR, style)
		images = os.listdir(style_path)
		for image in images:
			image_to_style[image] = style

	return image_to_style

if __name__ == '__main__':
	main()